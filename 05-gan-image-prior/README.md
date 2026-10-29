# GAN image priors for reconstruction

Experiments using pretrained **StyleGAN2** as a generative image prior for reconstruction and restoration. The analysis considers latent-space inversion and reconstruction under degradation, including blur, grayscale information loss, and masking.

## Technical focus

- Optimizing latent representations to reconstruct image observations.
- Evaluating restoration under different degradation operators.
- Comparing qualitative output and reconstruction behavior.

[Experiment report (PDF)](report.pdf)

## Implementation availability

The original experiments used a Colab notebook with external pretrained generators and starter components. Those dependencies are not bundled as an independently runnable project, so this directory contains the experimental report rather than a standalone training/inversion implementation.

**Skills:** generative image priors, latent optimization, image restoration, qualitative evaluation.
