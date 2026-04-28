# emission-line-net

A convolutional neural network for detection and characterisation of H&alpha; and [NII] emission lines in IFU spectroscopic data cubes. Designed for ram-pressure stripped galaxies where emission is present both in the galaxy disk and in the stripped tail, often at low signal-to-noise.

The network is trained entirely on synthetic spectra generated from a physically motivated noise model calibrated to each observed cube, requiring no labelled real data.

---

## Overview

The pipeline has three stages:

1. **Noise calibration** — estimates the Poisson noise scaling factor &alpha; from the observed flux and variance cubes by fitting Var = &sigma;<sub>sky</sub>&sup2; + &alpha; &middot; Flux across all spaxels.

2. **Synthetic training data generation** — generates spectra covering H&alpha; (&lambda;6562.8) and the [NII] doublet (&lambda;&lambda;6548, 6583) from five physically motivated regimes: background sky, ram-pressure stripped tail, disk without emission, disk with emission, and bright broad-line nuclear regions. The noise model matches the observed cube.

3. **Neural network training and inference** — a 1D CNN is trained jointly on binary line detection (classification) and parameter estimation (regression). At inference time it produces per-spaxel detection probabilities and maps of H&alpha; amplitude, [NII]/H&alpha; ratio, line-of-sight velocity, velocity dispersion, and continuum level.

---

## Model architecture

The network (`LineDetector`) uses a shared convolutional encoder with two independent heads:

- **Encoder**: two `Conv1d` layers (128 filters, kernel size 81) followed by global average pooling, producing a 256-element feature vector per spaxel.
- **Classification head**: maps to a single logit; sigmoid gives the line detection probability.
- **Regression head**: maps to 5 outputs — log&sub;10;(amp\_HA), log&sub;10;(NII/H&alpha;), vel, log&sub;10;(&sigma;<sub>obs</sub>), continuum.

The training loss has three components:

```
loss = clf_loss + λ_reg · reg_loss + λ_shape · shape_loss
```

where `shape_loss` is an IVAR-weighted MSE between the reconstructed model spectrum and the noiseless target, split into line and continuum regions to prevent amplitude-continuum degeneracy. Model weights from the best validation AUC epoch are restored before saving (checkpointing).

---

## Installation

```bash
pip install torch torchvision numpy scipy astropy scikit-learn matplotlib
```

For GPU support, install PyTorch with the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). The code automatically uses CUDA if available.

---

## Usage

```bash
python halpha_nn.py <galaxy_id> <redshift> [--force_train]
```

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `galaxy_id` | int | Galaxy ID used to locate the data cube at `clifs<id>/calibrated_cube.fits` |
| `redshift` | float | Spectroscopic redshift, used to shift the wavelength array to the rest frame |
| `--force_train` | flag | Force retraining even if a saved model exists |

**Example:**

```bash
python halpha_nn.py 8 0.024 --force_train
```

The script will:
1. Load and pre-process the observed cube
2. Estimate &alpha; and &sigma;<sub>sky</sub> from the data
3. Generate synthetic training (250k) and validation (50k) spectra
4. Train the network and save weights to `halpha_nn_model_regr.pt`
5. Run inference on every spaxel in the cube
6. Produce a detection probability map, binary detection map, five regression maps, and a 4&times;4 grid of spectral fit plots

On subsequent runs without `--force_train`, the saved model is loaded and only inference is performed.

---

## Input data format

The pipeline expects a FITS file at `clifs<id>/calibrated_cube.fits` with two extensions:

- `FLUX` — 3D flux cube, shape `(n_wav, ny, nx)`, with a standard 3D WCS header
- `IVAR` — inverse variance cube, same shape as FLUX

---

## Outputs

**Detection maps** (2-panel figure):
- Per-spaxel detection probability P(line) in [0, 1]
- Binary detection map at a user-defined threshold (default 0.75)

**Regression maps** (2&times;3 figure, masked to detected spaxels):
- H&alpha; amplitude [flux units]
- [NII] 6583 / H&alpha; amplitude ratio
- Line-of-sight velocity [km/s]
- Observed velocity dispersion [km/s] (LSF-convolved, floored at instrumental &sigma;)
- Continuum level [flux units]

**Spectral fit plots** (4&times;4 grid): randomly selected detected spaxels with the observed spectrum overlaid by the best-fit model and individual Gaussian components.

---

## Key design decisions

**Synthetic training data.** The network is trained entirely on synthetic spectra, avoiding the need for labelled real data. The noise model (sky + Poisson) is calibrated from the observed cube via `estimate_alpha`, so the synthetic noise matches the real instrument.

**Five training regimes.** Spectra are drawn from physically motivated regimes — background, stripped tail, disk without lines, disk with lines, and bright broad nuclear lines — to ensure coverage of the full parameter space encountered in ram-pressure stripping observations. The `bright_broad` regime explicitly oversamples the high-amplitude + high-dispersion corner that is rare under independent log-uniform sampling.

**Observed velocity dispersion.** The network regresses the observed (LSF-convolved) velocity dispersion rather than the intrinsic value. Below the instrumental floor (&sigma;<sub>instr</sub> = c / (2.355 &middot; R) &asymp; 51 km/s for R = 2500), intrinsic dispersion is unresolvable — predicting the observed quantity is the more tractable and physically honest target.

**IVAR-weighted shape loss.** The spectral shape loss weights each pixel by its inverse variance, downweighting noisy channels so that spurious noise spikes do not bias the amplitude or width estimates. The loss is split into separate line and continuum regions (`&lambda;_cont = 10`) to prevent the network from trading continuum errors against line amplitude.

**Model checkpointing.** The model state corresponding to the best validation AUC epoch is restored before saving, rather than using the final epoch weights.

---

## Configuration

Key hyperparameters are set at the top of the script or in `__main__`:

| Parameter | Default | Description |
|---|---|---|
| `HA_WAV` | 6562.819 Å | H&alpha; rest wavelength |
| `NII_WAV` | (6548.050, 6583.460) Å | [NII] doublet rest wavelengths |
| `NII_DOUBLET_RATIO` | 3.05 | Fixed [NII] 6583/6548 amplitude ratio |
| `WAVE_WINDOW` | 150 Å | Rest-frame spectral window half-width around H&alpha; |
| `n_filters` | 128 | Number of convolutional filters in first layer |
| `kernel_size` | 81 | Convolutional kernel size [pixels] |
| `train_size` | 250,000 | Number of training spectra |
| `val_size` | 50,000 | Number of validation spectra |
| `epochs` | 20 | Training epochs |
| `lambda_reg` | 1.0 | Regression loss weight |
| `lambda_shape` | 0.25 | Spectral shape loss weight |
| `lambda_cont` | 10.0 | Continuum region upweighting in shape loss |

---

## Known limitations

- **Single Gaussian model.** The current architecture fits one kinematic component per spaxel. Spaxels with two overlapping components (e.g. disk + stripped tail at the interface region) will be detected but poorly characterised; these can be identified by the shape loss residual being large. Multi-component fitting is planned for a future branch.
- **Fixed [NII] doublet ratio.** The [NII] 6583/6548 ratio is fixed at 3.05 by atomic physics. In practice, measurement noise means the two lines may not be individually detected at this ratio, particularly at low SNR.
- **Single-galaxy calibration.** The noise model (&alpha;, &sigma;<sub>sky</sub>) is estimated per cube. A model trained on one cube can be applied to others with different noise levels by retraining with `--force_train` after loading a new cube.

---

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- NumPy
- SciPy
- Astropy
- scikit-learn
- Matplotlib
