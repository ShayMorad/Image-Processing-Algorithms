# Gaussian/Laplacian pyramids and hybrid images

Multiscale image processing using Gaussian pyramids, Laplacian residuals, and mask-based image blending. A separate frequency-domain experiment constructs hybrid images by combining low- and high-frequency image content.

## Core functions

| Function | Purpose |
|---|---|
| `construct_gaus_pyr` | Gaussian pyramid construction |
| `construct_lap_pyr` | Laplacian pyramid decomposition |
| `construct_blended_lap_pyr` | Multiscale blending under a mask |
| `blend_images` | Reconstruction from blended pyramid levels |
| `compute_fourier` | Frequency-domain representation |
| `create_hybrid` | Low-/high-frequency hybrid image generation |

**Source:** [`ex3.py`](ex3.py) · [Technical report](report.pdf)

## Inputs and execution

The script's existing main path expects local images named `cat.jpg`, `lion.jpg`, `mask.jpg`, `cute_cat.jpg`, and `cute_dog.jpg`. These inputs are not included. Provide suitably licensed replacement images and a compatible mask before running the example. Outputs are written in the working directory.

**Skills:** scale-space representations, linear filtering, Fourier methods, numerical image reconstruction, multiresolution blending.
