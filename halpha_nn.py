import argparse
import os

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
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
        - vdisp_true    : intrinsic velocity dispersion [km/s]
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
    physics. The instrumental broadening is approximated as a Gaussian with
    sigma = c / (2.355 * R) km/s, and added in quadrature with vdisp_true.
    """

    rng = np.random.default_rng(seed=seed)

    if eline_params is None:
        noiseless = np.full(lam.shape, continuum)
    else:
        amp_HA, ha_nii_ratio, vel, vdisp_true = eline_params
        vdisp_instr = C_KMS / (2.355 * R)
        vdisp = np.sqrt(vdisp_true**2 + vdisp_instr**2)

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

    return lam, noiseless + sky_noise + poisson_noise

def sample_eline_params(rng, amp_HA=None, vdisp_range=(10, 300)):
    if amp_HA is None:
        amp_HA = 10 ** rng.uniform(np.log10(0.1), np.log10(30.0))
    ha_nii_ratio = 10 ** rng.uniform(np.log10(0.5), np.log10(10.0))
    vel          = rng.uniform(-300, 300)
    vdisp_true   = 10 ** rng.uniform(np.log10(vdisp_range[0]),
                                     np.log10(vdisp_range[1]))
    return (amp_HA, ha_nii_ratio, vel, vdisp_true)

def generate_dataset(n_spectra, lam, alpha=0.0,
                     regime_probs=(0.4, 0.25, 0.10, 0.25), seed=42):
    """
    Generate a dataset of synthetic spectra for binary line detection.

    Spectra are drawn from one of four physically motivated regimes:

      - 'background'  : off-disk, non-tail sky -- no continuum, no lines.
      - 'tail'        : ram-pressure stripped tail -- no continuum, faint
                        lines with narrow velocity dispersions (10-60 km/s).
      - 'disk_no_line': galaxy disk without detectable line emission --
                        positive continuum, no lines.
      - 'disk_line'   : galaxy disk with line emission -- positive continuum,
                        amp_HA loosely coupled to continuum via an equivalent
                        width ratio to reflect the observed correlation between
                        stellar continuum and line flux in star-forming regions.

    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate.
    alpha : float, optional
        Poisson noise scaling factor passed to mock_spectrum. Default 0.0.
    regime_probs : tuple of float, optional
        Sampling probabilities for
        (background, tail, disk_no_line, disk_line). Must sum to 1.
        Default (0.15, 0.20, 0.20, 0.45).
    seed : int, optional
        Master random seed. Default 42.

    Returns
    -------
    spectra : ndarray, shape (n_spectra, n_pixels)
    labels  : ndarray, shape (n_spectra,) -- 1 if lines present, 0 otherwise
    params  : list of tuple or None -- eline_params per spectrum
    lam     : ndarray, shape (n_pixels,) -- wavelength array [Angstroms]
    """
    rng = np.random.default_rng(seed)
    spectra, labels, params = [], [], []

    regimes = ["background", "tail", "disk_no_line", "disk_line"]

    for i in range(n_spectra):
        noise_sigma = 10 ** rng.uniform(np.log10(0.01), np.log10(0.5))
        regime = rng.choice(regimes, p=regime_probs)

        if regime == "background":
            # off-disk, non-tail: no continuum, no lines
            continuum    = rng.uniform(-0.1, 0.1)
            eline_params = None

        elif regime == "tail":
            # stripped tail: no continuum, faint lines, narrow vdisp
            continuum = rng.uniform(-0.1, 0.1)
            amp_HA    = 10 ** rng.uniform(np.log10(0.1), np.log10(2.0))
            eline_params = sample_eline_params(rng, amp_HA=amp_HA,
                                               vdisp_range=(10, 60))

        elif regime == "disk_no_line":
            # disk spaxel without detectable emission
            continuum    = 10 ** rng.uniform(np.log10(0.1), np.log10(10.0))
            eline_params = None

        else:  # disk_line
            # disk spaxel with emission -- amp_HA coupled to continuum via
            # an equivalent width ratio (physically ~10-100 AA for SF galaxies)
            continuum     = 10 ** rng.uniform(np.log10(0.1), np.log10(10.0))
            ha_cont_ratio = 10 ** rng.uniform(np.log10(0.1), np.log10(2.0))
            amp_HA        = continuum * ha_cont_ratio
            eline_params  = sample_eline_params(rng, amp_HA=amp_HA)

        _, spec = mock_spectrum(
            lam, continuum, noise_sigma,
            eline_params=eline_params,
            alpha=alpha,
            seed=int(rng.integers(1e6))
        )

        spectra.append(spec)
        labels.append(int(eline_params is not None))
        params.append(eline_params)       # None for no-line spectra

    return np.array(spectra), np.array(labels), params, lam

class SpectraDataset(Dataset):
    def __init__(self, spectra, labels):
        self.X = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LineDetector(nn.Module):
    def __init__(self, n_pixels, n_filters=32, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(n_filters, n_filters*2, kernel_size, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters*2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.conv(x)).squeeze(1)

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

def train_model(train_dataset, val_dataset, model, epochs=20,

        batch_size=256, lr=1e-3, device="cpu"):

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(epochs):
        # --- train ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimiser.step()

        # --- validate ---
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(device)).cpu()
                all_logits.append(logits)
                all_labels.append(y_batch)

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        auc = roc_auc_score(labels, logits)
        best_auc = max(best_auc, auc)
        print(f"  epoch {epoch+1:2d}/{epochs}  AUC={auc:.4f}")

    return model, best_auc

def learning_curve_experiment(alpha, wave, val_size=50_000, train_sizes=None,
        seed=42):

    if train_sizes is None:
        train_sizes = [10_000, 50_000, 200_000, 500_000, 1_000_000]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # fixed validation set - generated once, never changes
    print(f"Generating validation set ({val_size:,} spectra)...")
    val_spectra, val_labels, _, lam = generate_dataset(
            val_size, wave,
            alpha=alpha, seed=seed,
    )
    val_dataset = SpectraDataset(val_spectra, val_labels)
    n_pixels = val_spectra.shape[1]

    # generate largest training set once, then slice it
    # ensures smaller sets are strict subsets
    max_n = max(train_sizes)
    print(f"Generating full training pool ({max_n:,} spectra)...")
    all_spectra, all_labels, _, _ = generate_dataset(
            max_n, wave,
            alpha=alpha, seed=seed+1,
    )

    aucs = []
    for n in train_sizes:
        print(f"\n--- Training on {n:,} spectra ---")
        train_dataset = SpectraDataset(all_spectra[:n], all_labels[:n])
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
    def __init__(self, args):
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
        self.flux = flux[wavmask]
        self.var = var[wavmask]
        self.wave = wave[wavmask].to(u.AA).value
        self.y, self.x = np.mgrid[0:flux.shape[1], 0:flux.shape[2]]

def run_inference(model, obs, device="cpu", batch_size=1024):
    """
    Run a trained model on all spaxels in an observed IFU cube.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model. Must be in eval mode or will be set to eval.
    obs : ObservedData
        Observed data object containing flux cube and spatial coordinates.
    device : str, optional
        Torch device string. Default "cpu".
    batch_size : int, optional
        Number of spaxels per inference batch. Default 1024.

    Returns
    -------
    prob_map : ndarray, shape (ny, nx)
        Per-spaxel detection probability in [0, 1].
    """
    ny, nx = obs.flux.shape[1], obs.flux.shape[2]

    # reshape (n_wav, ny, nx) -> (ny*nx, 1, n_wav) for Conv1d
    spectra = obs.flux.transpose(1, 2, 0).reshape(-1, obs.flux.shape[0])
    spectra = spectra.astype(np.float32)
    spectra = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)

    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch  = spectra[i:i+batch_size].to(device)
            logits = model(batch).cpu()
            probs  = torch.sigmoid(logits)
            all_probs.append(probs)

    prob_map = torch.cat(all_probs).numpy().reshape(ny, nx)
    return prob_map

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cid", type=int)
    parser.add_argument("z", type=float)
    parser.add_argument("--force_train", action="store_true")
    args = parser.parse_args()

    obs = ObservedData(args)
    alpha, sigma_sky, result = estimate_alpha(obs.flux, obs.var,
        flux_max=1e+3, var_max=1e+3)

    seed = 42
    val_size = 50_000
    train_size = 250_000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = "halpha_nn_model.pt"
    n_pixels = obs.flux.shape[0]
    model = LineDetector(n_pixels, n_filters=128, kernel_size=81)
    if os.path.exists(model_path) and not args.force_train:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Generating validation set ({val_size:,} spectra)...")
        val_spectra, val_labels, _, lam = generate_dataset(
            val_size, obs.wave,
            alpha=alpha, seed=seed
        )
        val_dataset = SpectraDataset(val_spectra, val_labels)

        print(f"Generating training set ({train_size:,} spectra)...")
        train_spectra, train_labels, _, _ = generate_dataset(
            train_size, obs.wave,
            alpha=alpha, seed=seed+1
        )
        train_dataset = SpectraDataset(train_spectra, train_labels)

        model, auc = train_model(train_dataset, val_dataset, model,
            device=device)
        print(f"\nValidation AUC: {auc:.4f}")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("\nRunning inference on observed cube...")
    prob_map = run_inference(model, obs, device=device)
    plot_detection_map(prob_map, threshold=0.95)
