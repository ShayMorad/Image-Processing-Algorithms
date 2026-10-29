# Image Processing Algorithms

Implementations and experiments in video analysis, digital signal processing, multiscale image reconstruction, and video mosaicing. The repository covers five distinct topics, with Python source, technical reports, and selected visual results.

**Stack:** Python · OpenCV · NumPy · SciPy · Matplotlib · MediaPy  
**Methods:** Histograms, STFT/FFT, Gaussian and Laplacian pyramids, SIFT, geometric alignment, RANSAC

## Selected results

### Video stabilization and mosaicing

![Video mosaic generated from aligned frames](04-video-mosaicing/results/panorama_result.png)

*Panorama assembled from video frames using feature-based motion estimation and mosaicing.*

| Scene-cut detection | Audio Fourier analysis |
|:---:|:---:|
| ![Scene-cut histogram distance scores](assets/scene-cut-deltas.png) | ![Watermark spectrogram experiment](assets/watermark-spectrogram.png) |
| Consecutive-frame histogram differences | Time-frequency representation of an audio watermark |

[Frame-histogram transition](assets/scene-cut-transition.png) · [STFT peak visualization](assets/watermark-stft-peaks.png)

## Projects

| Topic | Main techniques | Code | Report |
|---|---|---|---|
| [01 · Scene-cut detection](01-scene-cut-detection/) | Frame histograms, cumulative histograms, scene-boundary scoring | [`ex1.py`](01-scene-cut-detection/ex1.py) | [PDF](01-scene-cut-detection/report.pdf) |
| [02 · Audio Fourier analysis](02-audio-fourier-analysis/) | Sinusoidal watermarks, STFT, spectral analysis, playback transformations | [`ex2.py`](02-audio-fourier-analysis/ex2.py) | [PDF](02-audio-fourier-analysis/report.pdf) |
| [03 · Pyramids and hybrid images](03-pyramids-and-hybrid-images/) | Gaussian/Laplacian pyramids, blending, spatial/frequency filters | [`ex3.py`](03-pyramids-and-hybrid-images/ex3.py) | [PDF](03-pyramids-and-hybrid-images/report.pdf) |
| [04 · Video mosaicing](04-video-mosaicing/) | SIFT, motion estimation, RANSAC, frame warping and mosaics | [`ex4.py`](04-video-mosaicing/ex4.py) | [PDF](04-video-mosaicing/report.pdf) |
| [05 · GAN image priors](05-gan-image-prior/) | StyleGAN2 inversion, reconstruction, image restoration experiments | [Experiment notes](05-gan-image-prior/) | [PDF](05-gan-image-prior/report.pdf) |

### Video mosaicing workflow

```mermaid
flowchart LR
    A[Video frames] --> B[Feature detection and matching]
    B --> C[Robust transform estimation]
    C --> D[Frame alignment / stabilization]
    D --> E[Strip selection and panorama composition]
```

The video-mosaicing directory contains multiple implementation variants exploring different alignment and stabilization strategies. See the [module README](04-video-mosaicing/) for a file-by-file explanation.

## Methods and skills

- **Computer vision:** feature extraction and correspondence, geometric model fitting, image warping, stabilization and panorama composition.
- **Digital signal processing:** discrete Fourier transforms, short-time spectra, watermark embedding/detection and frequency-domain filtering.
- **Multiscale methods:** Gaussian/Laplacian pyramids, mask-weighted blending, hybrid-image construction.
- **Numerical experimentation:** parameter comparison, diagnostic plots, reconstruction analysis, and documented results.

## Running the code

Create a Python environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Each experiment has a separate entry point and specific input media requirements. Video/audio/image datasets from the original runs are not bundled; substitute compatible, legally usable inputs and adjust the existing file paths where necessary. Module READMEs describe the expected files and commands.

The provided reports document the experimental setups and analyses. Some pipelines rely on fixed filenames and have not been verified end-to-end on a fresh machine.

## Repository structure

```text
01-scene-cut-detection/        Video frame histogram analysis
02-audio-fourier-analysis/     Audio watermarking and spectral processing
03-pyramids-and-hybrid-images/ Multiscale image synthesis and filtering
04-video-mosaicing/           Stabilization and panorama algorithms
05-gan-image-prior/           Generative-prior reconstruction experiments (report)
assets/                       Selected analysis figures
```

The fifth project is represented by its technical report; its notebook depends on external StyleGAN2 and exercise-specific starter components and is not included as a standalone implementation.
