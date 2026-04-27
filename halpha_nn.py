import argparse
import copy
import os

from astropy.io import fits
from astropy.nddata import block_reduce
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HA_WAV = 6562.819
NII_WAV = (6548.050, 6583.460)
C_KMS = 2.998e+5
NII_DOUBLET_RATIO = 3.05
WAVE_WINDOW = 150 # angstroms

def estimate_alpha(flux_cube, var_cube, flux_min=0.0, flux_max=None,
                   var_max=None, sigma_clip=3.0):
    """
    Estimate the Poisson noise scaling factor alpha from an IFU data cube.

    Fits the linear relation:
        Var = sigma_sky^2 + alpha * Flux
    across all spaxels and wavelength pixels, where alpha encodes the
    conversion between flux units and photon counts (~ inverse gain).

    Parameters
    ----------
    flux_cube : ndarray, shape (n_wav, ny, nx)
        Flux data cube.
    var_cube : ndarray, shape (n_wav, ny, nx)
        Variance data cube, same shape as flux_cube.
    flux_min : float, optional
        Minimum flux value to include. Default 0 (excludes negative flux).
    flux_max : float, optional
        Maximum flux value to include. Useful for excluding saturated pixels.
        Default None (no upper clip).
    sigma_clip : float, optional
        Sigma threshold for iterative outlier rejection. Default 3.0.

    Returns
    -------
    alpha : float
        Poisson noise scaling factor [variance / flux].
    sigma_sky : float
        Estimated sky noise standard deviation (sqrt of the intercept).
    result : scipy.stats.LinregressResult
        Full regression result for diagnostics.
    """
    flux_flat = flux_cube.ravel()
    var_flat  = var_cube.ravel()

    # --- basic mask ---
    mask = np.isfinite(flux_flat) & np.isfinite(var_flat) & (flux_flat > flux_min)
    mask &= flux_flat > 3*np.sqrt(var_flat)
    if flux_max is not None:
        mask &= flux_flat < flux_max
    if var_max is not None:
        mask &= var_flat < var_max

    flux_m = flux_flat[mask]
    var_m  = var_flat[mask]

    # --- iterative sigma clipping ---
    for _ in range(5):
        result = stats.linregress(flux_m, var_m)
        var_pred = result.slope * flux_m + result.intercept
        residuals = var_m - var_pred
        rms = np.std(residuals)
        keep = np.abs(residuals) < sigma_clip * rms
        if keep.sum() == len(flux_m):
            break
        flux_m, var_m = flux_m[keep], var_m[keep]

    alpha     = result.slope
    sigma_sky = np.sqrt(max(result.intercept, 0.0))  # guard against small negatives

    print(f"alpha     = {alpha:.4f}")
    print(f"sigma_sky = {sigma_sky:.4f}")
    print(f"R^2       = {result.rvalue**2:.4f}")
    print(f"N pixels  = {len(flux_m):,}")

    return alpha, sigma_sky, result

def plot_alpha_fit(flux_cube, var_cube, alpha, sigma_sky, result,
                   flux_max=None, var_max=None, max_points=10000):
    flux_flat = flux_cube.ravel()
    var_flat  = var_cube.ravel()
    mask = np.isfinite(flux_flat) & np.isfinite(var_flat) & (flux_flat > 0)
    if flux_max is not None:
        mask &= flux_flat < flux_max
    if var_max is not None:
        mask &= var_flat < var_max

    # subsample for plotting
    idx = np.random.choice(mask.sum(), min(max_points, mask.sum()), replace=False)
    f = flux_flat[mask][idx]
    v = var_flat[mask][idx]

    f_line = np.linspace(0, f.max(), 100)
    v_line = result.slope * f_line + result.intercept

    fig, ax = plt.subplots()
    ax.scatter(f, v, s=1, alpha=0.3, label='data')
    ax.plot(f_line, v_line, 'r-', label=f'α={alpha:.3f}, σ_sky={sigma_sky:.3f}')
    ax.set_xlabel('Flux')
    ax.set_ylabel('Variance')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

def _gauss(x, A, mu, sig):
    return A * np.exp(-(x - mu)**2 / (2*sig**2))


def build_model_spectra(reg, lam_t, R=2500):
    """
    Differentiable reconstruction of noiseless model spectra from regression
    parameters. Used in the shape loss to penalise incorrect line morphology.

    Parameters
    ----------
    reg : tensor, shape (batch, 5)
        Regression parameters [log10(amp_HA), log10(nii_ha), vel,
        log10(vdisp_obs), continuum]. First 4 may be NaN for no-line examples.
    vdisp_obs is the observed (LSF-convolved) dispersion, floored at the
    instrumental sigma.
    lam_t : tensor, shape (n_pixels,)
        Wavelength array [Angstroms], on the same device as reg.
    R : float, optional
        Instrumental spectral resolution. Default 2500.

    Returns
    -------
    spectra : tensor, shape (batch, n_pixels)
        Reconstructed noiseless spectra. No-line examples contain only the
        continuum level; line examples contain continuum + Gaussians.
    """
    cont = reg[:, 4].unsqueeze(1)          # (batch, 1)
    lam  = lam_t.unsqueeze(0)              # (1, n_pixels)

    # start with flat continuum for all examples
    spectra = cont.expand(-1, lam_t.shape[0]).clone()

    # identify positive examples — NaN in first param flags no-line spectra
    has_lines = ~torch.isnan(reg[:, 0])

    if has_lines.any():
        r = reg[has_lines]
        amp_HA  = 10 ** r[:, 0]
        nii_ha  = 10 ** r[:, 1]
        vel     = r[:, 2]
        vdisp_t = 10 ** r[:, 3]

        # convolve with instrumental LSF
        # vdisp_t is already the observed (LSF-convolved) dispersion
        # shifted line centres
        lam_HA    = HA_WAV     * (1 + vel / C_KMS)
        lam_NII_1 = NII_WAV[0] * (1 + vel / C_KMS)
        lam_NII_2 = NII_WAV[1] * (1 + vel / C_KMS)

        # wavelength sigmas — vdisp_t is observed, floor at 1e-2 AA
        sig_HA    = (HA_WAV     * vdisp_t / C_KMS).clamp(min=1e-2)
        sig_NII_1 = (NII_WAV[0] * vdisp_t / C_KMS).clamp(min=1e-2)
        sig_NII_2 = (NII_WAV[1] * vdisp_t / C_KMS).clamp(min=1e-2)

        # line amplitudes
        amp_NII_2 = amp_HA / nii_ha
        amp_NII_1 = amp_NII_2 / NII_DOUBLET_RATIO

        def gauss_t(A, mu, sig):
            # broadcast (batch, 1) against (1, n_pixels)
            return (
                A.unsqueeze(1)
                * torch.exp(-(lam - mu.unsqueeze(1))**2
                            / (2 * sig.unsqueeze(1)**2))
            )

        signal = (
            gauss_t(amp_HA,    lam_HA,    sig_HA)
            + gauss_t(amp_NII_1, lam_NII_1, sig_NII_1)
            + gauss_t(amp_NII_2, lam_NII_2, sig_NII_2)
        )
        spectra[has_lines] = spectra[has_lines] + signal

    return spectra

def mock_spectrum(lam, continuum, noise_sigma, eline_params=None,
        alpha=0.0, R=2500, seed=42):
    """
    Generate a synthetic spectrum centred on H-alpha and the NII doublet.

    The spectrum consists of an optional emission-line signal (three Gaussians)
    added to a flat continuum and Gaussian noise. The observed line width is the
    quadrature sum of the intrinsic velocity dispersion and the instrumental LSF.

    The total noise model is:
        sigma_total(lambda)^2 = noise_sigma^2 + alpha * max(f(lambda), 0)
    where the first term is the (constant) sky background noise and the second
    is a pixel-dependent Poisson term that scales with the local flux.

    Parameters
    ----------
    continuum : float
        Flat continuum level added to the spectrum.
    noise_sigma : float
        Standard deviation of the sky background noise (Gaussian, per pixel).
    eline_params : tuple of float, optional
        Emission line parameters (amp_HA, ha_nii_ratio, vel, vdisp_true).
        - amp_HA        : H-alpha line amplitude [flux units]
        - ha_nii_ratio  : H-alpha to NII 6583 amplitude ratio
        - vel           : line-of-sight velocity shift [km/s]
        - vdisp_true    : observed velocity dispersion [km/s],
                          floored at the instrumental sigma
        If None, returns a pure noise + continuum spectrum.
    alpha : float, optional
        Poisson noise scaling factor [variance / flux], as estimated from the
        data cube via estimate_alpha(). Default 0.0 (no Poisson noise).
    R : float, optional
        Instrumental spectral resolution (lambda / d_lambda). Default 2500.
    seed : int, optional
        Random seed for reproducibility. Default 42.

    Returns
    -------
    lam : ndarray
        Wavelength array [Angstroms].
    spectrum : ndarray
        Synthetic spectrum [flux units], same shape as lam.

    Notes
    -----
    The NII doublet amplitude ratio (6583/6548) is fixed at 3.05 by atomic
    physics. vdisp_true is treated as the observed (LSF-convolved) line
    width, floored at the instrumental sigma = c / (2.355 * R) km/s.
    """

    rng = np.random.default_rng(seed=seed)

    if eline_params is None:
        noiseless = np.full(lam.shape, continuum)
    else:
        amp_HA, ha_nii_ratio, vel, vdisp_true = eline_params
        # vdisp_true is now the observed dispersion (already LSF-convolved)
        vdisp = vdisp_true

        lam_NII_1 = NII_WAV[0] * (1 + vel/C_KMS)
        lam_HA = HA_WAV * (1 + vel/C_KMS)
        lam_NII_2 = NII_WAV[1] * (1 + vel/C_KMS)

        lam_disp_NII_1 = NII_WAV[0] * vdisp / C_KMS
        lam_disp_HA = HA_WAV * vdisp / C_KMS
        lam_disp_NII_2 = NII_WAV[1] * vdisp / C_KMS

        amp_NII_1 = amp_HA / ha_nii_ratio / NII_DOUBLET_RATIO
        amp_NII_2 = amp_HA / ha_nii_ratio

        signal = (
            _gauss(lam, amp_NII_1, lam_NII_1, lam_disp_NII_1)
            + _gauss(lam, amp_HA, lam_HA, lam_disp_HA)
            + _gauss(lam, amp_NII_2, lam_NII_2, lam_disp_NII_2)
        )
        noiseless = signal + continuum

    sky_noise = rng.normal(0, noise_sigma, lam.shape)
    poisson_sigma = np.sqrt(alpha * np.maximum(noiseless, 0.0))
    poisson_noise = rng.normal(0, poisson_sigma, lam.shape)

    # IVAR: inverse of total variance, pixel-dependent
    var_total = noise_sigma**2 + alpha * np.maximum(noiseless, 0.0)
    ivar = np.where(var_total > 0, 1.0 / var_total, 0.0)

    return lam, noiseless + sky_noise + poisson_noise, ivar

def sample_eline_params(rng, amp_HA=None, vdisp_range=(10, 300),
                         R=2500):
    if amp_HA is None:
        amp_HA = 10 ** rng.uniform(np.log10(0.1), np.log10(400.0))
    ha_nii_ratio = 10 ** rng.uniform(np.log10(0.5), np.log10(10.0))
    vel          = rng.uniform(-300, 300)
    vdisp_obs    = 10 ** rng.uniform(np.log10(vdisp_range[0]),
                                     np.log10(vdisp_range[1]))
    # floor at instrumental dispersion — observed sigma cannot be smaller
    vdisp_instr  = C_KMS / (2.355 * R)
    vdisp_obs    = max(vdisp_obs, vdisp_instr)
    return (amp_HA, ha_nii_ratio, vel, vdisp_obs)

def generate_dataset(n_spectra, lam, alpha=0.0,
                     regime_probs=(0.35, 0.23, 0.09, 0.23, 0.10), seed=42):
    """
    Generate a dataset of synthetic spectra for binary line detection.

    Spectra are drawn from one of five physically motivated regimes:

      - 'background'  : off-disk, non-tail sky -- no continuum, no lines.
      - 'tail'        : ram-pressure stripped tail -- no continuum, faint
                        lines with narrow velocity dispersions (10-60 km/s).
      - 'disk_no_line': galaxy disk without detectable line emission --
                        positive continuum, no lines.
      - 'disk_line'   : galaxy disk with line emission -- positive continuum,
                        amp_HA loosely coupled to continuum via an equivalent
                        width ratio to reflect the observed correlation between
                        stellar continuum and line flux in star-forming regions.
      - 'bright_broad': high-amplitude, high-vdisp lines -- oversample the
                        corner of parameter space where both amp_HA and
                        vdisp_obs are near their upper limits, which is
                        otherwise rare under independent log-uniform sampling.

    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate.
    alpha : float, optional
        Poisson noise scaling factor passed to mock_spectrum. Default 0.0.
    regime_probs : tuple of float, optional
        Sampling probabilities for
        (background, tail, disk_no_line, disk_line, bright_broad).
        Must sum to 1. Default (0.35, 0.23, 0.09, 0.23, 0.10).
    seed : int, optional
        Master random seed. Default 42.

    Returns
    -------
    spectra     : ndarray, shape (n_spectra, n_pixels)
    ivars       : ndarray, shape (n_spectra, n_pixels) -- inverse variance per pixel
    labels      : ndarray, shape (n_spectra,) -- 1 if lines present, 0 otherwise
    reg_targets : ndarray, shape (n_spectra, 5) -- regression targets
                  [log10(amp_HA), log10(ha_nii_ratio), vel, log10(vdisp_obs),
                   continuum]. First 4 are NaN for no-line spectra;
                  continuum is always set. vdisp_obs is the observed
                  dispersion, floored at the instrumental sigma.
    params      : list of tuple or None -- eline_params per spectrum
    lam         : ndarray, shape (n_pixels,) -- wavelength array [Angstroms]
    """
    rng = np.random.default_rng(seed)
    spectra, ivars, labels, reg_targets, params = [], [], [], [], []

    regimes = ["background", "tail", "disk_no_line", "disk_line",
               "bright_broad"]

    for i in range(n_spectra):
        noise_sigma = 10 ** rng.uniform(np.log10(0.01), np.log10(0.5))
        regime = rng.choice(regimes, p=regime_probs)

        if regime == "background":
            # off-disk, non-tail: no continuum, no lines
            continuum    = rng.uniform(-0.5, 0.1)
            eline_params = None

        elif regime == "tail":
            # stripped tail: no continuum, faint lines, narrow vdisp
            continuum = rng.uniform(-0.5, 0.1)
            amp_HA    = 10 ** rng.uniform(np.log10(0.1), np.log10(2.0))
            eline_params = sample_eline_params(rng, amp_HA=amp_HA,
                                               vdisp_range=(10, 60))

        elif regime == "disk_no_line":
            # disk spaxel without detectable emission
            continuum    = 10 ** rng.uniform(np.log10(0.1), np.log10(10.0))
            eline_params = None

        elif regime == "disk_line":
            # disk spaxel with emission -- continuum mostly positive but
            # allow near-zero/negative cases for outer disk / tail interface
            if rng.random() < 0.15:
                continuum = rng.uniform(-0.5, 0.1)
            else:
                continuum = 10 ** rng.uniform(np.log10(0.1), np.log10(10.0))
            amp_HA       = 10 ** rng.uniform(np.log10(0.1), np.log10(400.0))
            eline_params = sample_eline_params(rng, amp_HA=amp_HA)

        else:  # bright_broad
            # oversample high-amplitude + high-vdisp corner — rare under
            # independent log-uniform sampling but present in bright nuclei
            continuum    = 10 ** rng.uniform(np.log10(1.0), np.log10(100.0))
            amp_HA       = 10 ** rng.uniform(np.log10(10.0), np.log10(400.0))
            eline_params = sample_eline_params(rng, amp_HA=amp_HA,
                                               vdisp_range=(150, 300))

        _, spec, ivar = mock_spectrum(
            lam, continuum, noise_sigma,
            eline_params=eline_params,
            alpha=alpha,
            seed=int(rng.integers(1e6))
        )

        spectra.append(spec)
        ivars.append(ivar)
        labels.append(int(eline_params is not None))
        params.append(eline_params)       # None for no-line spectra

        # regression targets: log10(amp_HA), log10(ha_nii_ratio),
        #                     vel [km/s],    log10(vdisp_obs),
        #                     continuum [flux units, linear]
        # First 4 are NaN for no-line spectra; continuum is always set.
        if eline_params is not None:
            amp_HA, ha_nii_ratio, vel, vdisp_obs = eline_params
            reg_target = np.array([
                np.log10(amp_HA),
                np.log10(ha_nii_ratio),
                vel,
                np.log10(vdisp_obs),
                continuum,
            ], dtype=np.float32)
        else:
            reg_target = np.array([
                np.nan, np.nan, np.nan, np.nan,
                continuum,
            ], dtype=np.float32)
        reg_targets.append(reg_target)

    return np.array(spectra), np.array(ivars), np.array(labels), np.array(reg_targets), params, lam

class SpectraDataset(Dataset):
    def __init__(self, spectra, ivars, labels, reg_targets):
        self.X    = torch.tensor(spectra,     dtype=torch.float32).unsqueeze(1)
        self.ivar = torch.tensor(ivars,       dtype=torch.float32)
        self.y    = torch.tensor(labels,      dtype=torch.float32)
        self.reg  = torch.tensor(reg_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.ivar[idx], self.y[idx], self.reg[idx]

class LineDetector(nn.Module):
    def __init__(self, n_pixels, n_filters=32, kernel_size=7):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(n_filters, n_filters*2, kernel_size, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        # classification head: outputs single logit
        self.clf_head = nn.Sequential(
            nn.Linear(n_filters*2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # regression head: outputs 5 values
        #   [log10(amp_HA), log10(ha_nii_ratio), vel, log10(vdisp_obs),
        #    continuum]  -- vdisp_obs is observed (LSF-convolved) dispersion
        self.reg_head = nn.Sequential(
            nn.Linear(n_filters*2, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        features = self.encoder(x)
        logit    = self.clf_head(features).squeeze(1)
        reg      = self.reg_head(features)
        return logit, reg

class LineDetectorMLP(nn.Module):
    def __init__(self, n_pixels, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_pixels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_model(train_dataset, val_dataset, model, lam, epochs=20,
        lambda_reg=1.0, lambda_shape=0.1, lambda_cont=10.0,
        batch_size=256, lr=1e-3, device="cpu"):
    """
    Train the LineDetector jointly on classification and regression tasks.

    Parameters
    ----------
    train_dataset, val_dataset : SpectraDataset
    model : LineDetector
    lam : ndarray, shape (n_pixels,)
        Wavelength array [Angstroms]. Used to construct model spectra
        for the shape loss.
    epochs : int
    lambda_reg : float
        Weight of the regression loss relative to classification loss.
        Default 1.0.
    lambda_shape : float
        Weight of the spectral shape loss. Penalises mismatch between
        the reconstructed model spectrum and the noiseless target,
        addressing amplitude-width degeneracy. Default 0.1.
    lambda_cont : float
        Additional upweighting for the continuum region of the shape
        loss, relative to the line region. Prevents the network from
        inflating line amplitude to compensate for continuum errors.
        Default 10.0.
    batch_size : int
    lr : float
    device : str

    Returns
    -------
    model : trained LineDetector
    best_auc : float
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model       = model.to(device)
    optimiser   = torch.optim.Adam(model.parameters(), lr=lr)
    clf_loss_fn = nn.BCEWithLogitsLoss()
    reg_loss_fn = nn.HuberLoss(reduction="none")
    lam_t       = torch.tensor(lam, dtype=torch.float32).to(device)

    # spectral masks for split shape loss
    # line region: covers full NII+HA complex with generous margin
    # continuum region: everything outside the line complex
    line_mask_t = torch.tensor(
        (lam > HA_WAV - 30) & (lam < HA_WAV + 30),
        dtype=torch.bool, device=device
    )
    cont_mask_t = ~line_mask_t

    best_auc   = 0.0
    best_state = None
    for epoch in range(epochs):
        # --- train ---
        model.train()
        for X_batch, ivar_batch, y_batch, reg_batch in train_loader:
            X_batch    = X_batch.to(device)
            ivar_batch = ivar_batch.to(device)
            y_batch    = y_batch.to(device)
            reg_batch  = reg_batch.to(device)

            # normalise IVAR to unit mean per batch so it acts as a
            # relative weight without changing the absolute loss scale
            ivar_w = ivar_batch / ivar_batch.mean().clamp(min=1e-6)

            optimiser.zero_grad()
            logits, reg_pred = model(X_batch)

            # classification loss over all examples
            clf_loss = clf_loss_fn(logits, y_batch)

            # line parameter loss: only on positive examples (label=1)
            # continuum loss: on all examples (always defined)
            pos_mask = y_batch.bool()
            if pos_mask.any():
                line_loss = reg_loss_fn(
                    reg_pred[pos_mask, :4],
                    reg_batch[pos_mask, :4]
                ).mean()
            else:
                line_loss = torch.tensor(0.0, device=device)
            cont_loss = reg_loss_fn(
                reg_pred[:, 4],
                reg_batch[:, 4]
            ).mean()
            reg_loss = line_loss + cont_loss

            # shape loss: penalise mismatch in reconstructed spectrum
            # targets built from ground-truth params; predictions from
            # reg_pred — forces correct line morphology, breaking the
            # amplitude-width degeneracy
            # clamp predictions to training range before reconstruction
            # to prevent overflow (e.g. 10**large) corrupting gradients
            reg_pred_clamped = torch.stack([
                reg_pred[:, 0].clamp(-1.0, 2.6),   # log10(amp_HA)
                reg_pred[:, 1].clamp(-0.3, 1.0),   # log10(nii_ha)
                reg_pred[:, 2].clamp(-400, 400),    # vel [km/s]
                reg_pred[:, 3].clamp( 0.7, 2.5),   # log10(vdisp)
                reg_pred[:, 4],                     # continuum
            ], dim=1)
            pred_spectra   = build_model_spectra(reg_pred_clamped, lam_t)
            target_spectra = build_model_spectra(reg_batch,        lam_t)

            # IVAR-weighted split shape loss: downweight noisy pixels
            # so spurious noise spikes don't inflate line amplitude
            residuals = (pred_spectra - target_spectra) ** 2
            shape_loss_line = (residuals[:, line_mask_t]
                               * ivar_w[:, line_mask_t]).mean()
            shape_loss_cont = (residuals[:, cont_mask_t]
                               * ivar_w[:, cont_mask_t]).mean()
            shape_loss = shape_loss_line + lambda_cont * shape_loss_cont

            loss = (clf_loss
                    + lambda_reg   * reg_loss
                    + lambda_shape * shape_loss)
            loss.backward()
            # clip gradients to prevent any bad batch corrupting weights
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0)
            optimiser.step()

        # --- validate ---
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for X_batch, _, y_batch, _ in val_loader:
                logits, _ = model(X_batch.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(y_batch)

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        auc = roc_auc_score(labels, logits)
        if auc > best_auc:
            best_auc   = auc
            best_state = copy.deepcopy(model.state_dict())
        print(f"  epoch {epoch+1:2d}/{epochs}  AUC={auc:.4f}  best={best_auc:.4f}")

    # restore weights from the best epoch before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc

def learning_curve_experiment(alpha, wave, val_size=50_000, train_sizes=None,
        seed=42):

    if train_sizes is None:
        train_sizes = [10_000, 50_000, 200_000, 500_000, 1_000_000]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # fixed validation set - generated once, never changes
    print(f"Generating validation set ({val_size:,} spectra)...")
    val_spectra, val_ivars, val_labels, val_reg, _, lam = generate_dataset(
            val_size, wave,
            alpha=alpha, seed=seed,
    )
    val_dataset = SpectraDataset(val_spectra, val_ivars, val_labels, val_reg)
    n_pixels = val_spectra.shape[1]

    # generate largest training set once, then slice it
    # ensures smaller sets are strict subsets
    max_n = max(train_sizes)
    print(f"Generating full training pool ({max_n:,} spectra)...")
    all_spectra, all_ivars, all_labels, all_reg, _, _ = generate_dataset(
            max_n, wave,
            alpha=alpha, seed=seed+1,
    )

    aucs = []
    for n in train_sizes:
        print(f"\n--- Training on {n:,} spectra ---")
        train_dataset = SpectraDataset(all_spectra[:n], all_ivars[:n], all_labels[:n], all_reg[:n])
        _, auc = train_model(
            train_dataset,
            val_dataset,
            n_pixels,
            device=device,
        )
        aucs.append(auc)
        print(f"Best AUC: {auc:.4f}")

    return train_sizes, aucs

def plot_learning_curve(train_sizes, aucs):
    fig, ax = plt.subplots()
    ax.plot(train_sizes, aucs, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Validation AUC")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

class ObservedData:
    def __init__(self, args, Nrebin=None):
        hdul = fits.open(f"clifs{args.cid}/calibrated_cube.fits")
        flux = hdul["FLUX"].data
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        var = 1/hdul["IVAR"].data
        self.hdr = hdul["FLUX"].header
        self.wcs = WCS(self.hdr)
        self.z = args.z
        wave = self.wcs.spectral.pixel_to_world(np.arange(flux.shape[0]))
        wave /= 1 + self.z
        wavmask = (
            (wave.to(u.AA).value > HA_WAV - WAVE_WINDOW//2)
            & (wave.to(u.AA).value < HA_WAV + WAVE_WINDOW//2)
        )
        self.wave = wave[wavmask].to(u.AA).value
        if Nrebin is None:
            self.flux = flux[wavmask]
            self.var = var[wavmask]
            self.y, self.x = np.mgrid[0:self.flux.shape[1], 0:self.flux.shape[2]]
        else:
            self.flux = block_reduce(flux[wavmask],
                block_size=(1, Nrebin, Nrebin), func=np.sum)
            self.var = block_reduce(var[wavmask],
                block_size=(1, Nrebin, Nrebin), func=np.sum)
            self.y, self.x = np.mgrid[0:self.flux.shape[1], 0:self.flux.shape[2]]

def run_inference(model, obs, device="cpu", batch_size=1024, R=2500):
    """
    Run a trained model on all spaxels in an observed IFU cube.

    Parameters
    ----------
    model : LineDetector
        Trained model.
    obs : ObservedData
        Observed data object.
    device : str, optional
        Default "cpu".
    batch_size : int, optional
        Default 1024.

    Returns
    -------
    prob_map : ndarray, shape (ny, nx)
        Per-spaxel detection probability in [0, 1].
    reg_maps : dict of ndarray, shape (ny, nx)
        Per-spaxel regression outputs, keyed by parameter name.
        Values are in physical units (amp_HA in flux units, vel in km/s,
        vdisp in km/s). Only meaningful where prob_map is high.
    """
    ny, nx = obs.flux.shape[1], obs.flux.shape[2]

    spectra = obs.flux.transpose(1, 2, 0).reshape(-1, obs.flux.shape[0])
    spectra = spectra.astype(np.float32)
    spectra = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)

    model = model.to(device)
    model.eval()

    all_probs, all_reg = [], []
    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch        = spectra[i:i+batch_size].to(device)
            logits, reg  = model(batch)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_reg.append(reg.cpu())

    prob_map = torch.cat(all_probs).numpy().reshape(ny, nx)
    reg_out  = torch.cat(all_reg).numpy()   # shape (ny*nx, 4)

    # convert from log/linear predicted space back to physical units
    reg_maps = {
        "amp_HA"    : (10 ** reg_out[:, 0]).reshape(ny, nx),
        "nii_ha"    : (10 ** reg_out[:, 1]).reshape(ny, nx),
        "vel"       :        reg_out[:, 2].reshape(ny, nx),
        "vdisp"     : (10 ** reg_out[:, 3]).reshape(ny, nx),
        "continuum" :        reg_out[:, 4].reshape(ny, nx),
    }
    # vdisp is already in observed units — no LSF convolution needed
    return prob_map, reg_maps

def plot_detection_map(prob_map, threshold=0.5):
    """
    Plot the per-spaxel detection probability map and binary detection map.

    Parameters
    ----------
    prob_map : ndarray, shape (ny, nx)
        Per-spaxel detection probabilities from run_inference().
    threshold : float, optional
        Probability threshold for binary detection. Default 0.5.
    """
    detection_map = (prob_map >= threshold).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(prob_map, origin="lower", cmap="viridis",
                          vmin=0, vmax=1)
    axes[0].set_title("Detection probability")
    axes[0].set_xlabel("x [pixels]")
    axes[0].set_ylabel("y [pixels]")
    plt.colorbar(im0, ax=axes[0], label="P(line)")

    im1 = axes[1].imshow(detection_map, origin="lower", cmap="binary_r",
                          vmin=0, vmax=1)
    axes[1].set_title(f"Binary detection map (threshold={threshold})")
    axes[1].set_xlabel("x [pixels]")
    axes[1].set_ylabel("y [pixels]")
    plt.colorbar(im1, ax=axes[1], label="Detection")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_spectral_fits(obs, prob_map, reg_maps, threshold=0.75,
                       n_plots=16, seed=42):
    """
    Plot observed spectra overlaid with best-fit Gaussian models for a random
    selection of spaxels with detected emission lines.

    Parameters
    ----------
    obs : ObservedData
        Observed data object containing flux cube and wavelength array.
    prob_map : ndarray, shape (ny, nx)
        Per-spaxel detection probabilities from run_inference().
    reg_maps : dict of ndarray, shape (ny, nx)
        Regression outputs from run_inference(), keyed by parameter name.
    threshold : float, optional
        Probability threshold for detection. Default 0.75.
    n_plots : int, optional
        Number of spaxels to plot. Must be a perfect square. Default 16.
    seed : int, optional
        Random seed for reproducible spaxel selection. Default 42.
    """
    rng = np.random.default_rng(seed)
    n_side = int(np.sqrt(n_plots))

    # find all detected spaxels and randomly select n_plots of them
    detected_y, detected_x = np.where(prob_map >= threshold)
    n_detected = len(detected_y)
    if n_detected < n_plots:
        print(f"Warning: only {n_detected} detected spaxels, "
              f"plotting all of them.")
        n_plots = n_detected
        n_side  = int(np.sqrt(n_plots))

    idx = rng.choice(n_detected, size=n_plots, replace=False)
    ys  = detected_y[idx]
    xs  = detected_x[idx]

    # fine wavelength grid for smooth model curves
    lam_fine = np.linspace(obs.wave.min(), obs.wave.max(), 1000)

    fig, axes = plt.subplots(n_side, n_side,
                             figsize=(3.5 * n_side, 2.5 * n_side))

    for ax, y, x in zip(axes.flat, ys, xs):
        spec = obs.flux[:, y, x]
        ax.step(obs.wave, spec, color="black", linewidth=0.8, where="mid",
                label="observed")

        # best-fit parameters
        amp_HA    = reg_maps["amp_HA"][y, x]
        nii_ha    = reg_maps["nii_ha"][y, x]
        vel       = reg_maps["vel"][y, x]
        vdisp     = reg_maps["vdisp"][y, x]   # already convolved with LSF
        cont      = reg_maps["continuum"][y, x]

        # wavelength sigma for each line
        sig_HA    = HA_WAV     * vdisp / C_KMS
        sig_NII_1 = NII_WAV[0] * vdisp / C_KMS
        sig_NII_2 = NII_WAV[1] * vdisp / C_KMS

        # shifted line centres
        lam_HA    = HA_WAV     * (1 + vel / C_KMS)
        lam_NII_1 = NII_WAV[0] * (1 + vel / C_KMS)
        lam_NII_2 = NII_WAV[1] * (1 + vel / C_KMS)

        # line amplitudes from nii_ha ratio
        amp_NII_2 = amp_HA / nii_ha
        amp_NII_1 = amp_NII_2 / NII_DOUBLET_RATIO

        # individual Gaussian components
        g_HA    = _gauss(lam_fine, amp_HA,    lam_HA,    sig_HA)
        g_NII_1 = _gauss(lam_fine, amp_NII_1, lam_NII_1, sig_NII_1)
        g_NII_2 = _gauss(lam_fine, amp_NII_2, lam_NII_2, sig_NII_2)
        model   = g_HA + g_NII_1 + g_NII_2

        # plot model and components, all offset by continuum
        ax.plot(lam_fine, model + cont,   color="red",    linewidth=1.2,
                label="model")
        ax.plot(lam_fine, g_HA + cont,    color="blue",   linewidth=0.8,
                linestyle="--", label=r"H$\alpha$")
        ax.plot(lam_fine, g_NII_1 + cont, color="green",  linewidth=0.8,
                linestyle="--", label="NII 6548")
        ax.plot(lam_fine, g_NII_2 + cont, color="purple", linewidth=0.8,
                linestyle="--", label="NII 6583")

        ax.set_title(f"({x}, {y})  p={prob_map[y,x]:.2f}", fontsize=8)
        ax.set_xlabel(r"$\lambda$ [$\AA$]", fontsize=7)
        ax.set_ylabel("Flux", fontsize=7)
        ax.tick_params(labelsize=6)

        ymax = max((model + cont).max(), spec.max()) * 1.2
        ymin = min(spec.min(), cont) * 1.1
        ax.set_ylim(ymin, ymax)

    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower right", fontsize=8,
               ncol=2, framealpha=0.9)

    plt.suptitle("Best-fit spectral models (random detected spaxels)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cid", type=int)
    parser.add_argument("z", type=float)
    parser.add_argument("--force_train", action="store_true")
    args = parser.parse_args()

    obs = ObservedData(args, Nrebin=2)
    alpha, sigma_sky, result = estimate_alpha(obs.flux, obs.var,
        flux_max=1e+3, var_max=1e+3)

    seed = 42
    val_size = 50_000
    train_size = 250_000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = "halpha_nn_model_regr.pt"
    n_pixels = obs.flux.shape[0]
    model = LineDetector(n_pixels, n_filters=128, kernel_size=81)
    if os.path.exists(model_path) and not args.force_train:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Generating validation set ({val_size:,} spectra)...")
        val_spectra, val_ivars, val_labels, val_reg, _, lam = generate_dataset(
            val_size, obs.wave,
            alpha=alpha, seed=seed
        )
        val_dataset = SpectraDataset(val_spectra, val_ivars, val_labels, val_reg)

        print(f"Generating training set ({train_size:,} spectra)...")
        train_spectra, train_ivars, train_labels, train_reg, _, _ = generate_dataset(
            train_size, obs.wave,
            alpha=alpha, seed=seed+1
        )
        train_dataset = SpectraDataset(train_spectra, train_ivars, train_labels, train_reg)

        model, auc = train_model(train_dataset, val_dataset, model,
            lam=lam, device=device, lambda_shape=0.25)
        print(f"\nValidation AUC: {auc:.4f}")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("\nRunning inference on observed cube...")
    threshold = 0.75
    prob_map, reg_maps = run_inference(model, obs, device=device)
    plot_detection_map(prob_map, threshold=threshold)

    # plot regression maps, masked to detected spaxels
    detected = prob_map >= threshold
    titles  = {"amp_HA":    "Hα amplitude [flux]",
               "nii_ha":    "Hα/NII ratio",
               "vel":       "Velocity [km/s]",
               "vdisp":     "Velocity dispersion [km/s]",
               "continuum": "Continuum [flux]"}
    cmaps   = {"amp_HA":    "inferno", "nii_ha":    "viridis",
               "vel":       "RdBu_r",  "vdisp":     "plasma",
               "continuum": "cividis"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title) in zip(axes.flat, titles.items()):
        data = np.where(detected, reg_maps[key], np.nan)
        if key in ("amp_HA", "continuum") and np.nanpercentile(data, 1) > 0:
            norm = colors.LogNorm(
                vmin=np.nanpercentile(data, 1),
                vmax=np.nanpercentile(data, 99),
            )
        else:
            norm = colors.Normalize(
                vmin=np.nanpercentile(data, 1),
                vmax=np.nanpercentile(data, 99),
            )
        im   = ax.imshow(data, origin="lower", cmap=cmaps[key], norm=norm)
        ax.set_title(title)
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
    plt.close()

    print("\nPlotting spectral fits...")
    plot_spectral_fits(obs, prob_map, reg_maps, threshold=threshold)
