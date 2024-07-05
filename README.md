# FORA
Diffusion transformers (DiT) have become the de facto choice for generating high-quality images and videos, largely due to their scalability, which enables the construction of larger models for enhanced performance. However, the increased size of these models leads to higher inference costs, making them less attractive for real-time applications. We present **F**ast-F**OR**ward C**A**ching (FORA), a simple yet effective approach designed to accelerate DiT by exploiting the repetitive nature of the diffusion process. FORA implements a caching mechanism that stores and reuses intermediate outputs from the attention and MLP layers across denoising steps, thereby reducing computational overhead. This approach does not require model retraining and seamlessly integrates with existing transformer-based diffusion models. Experiments show that FORA can speed up diffusion transformers several times over while only minimally affecting performance metrics such as the IS Score and FID. By enabling faster processing with minimal trade-offs in quality, FORA represents a significant advancement in deploying diffusion transformers for real-time applications. Code will be made publicly available.

![Teaser](FORA_teaser.png)

## Acknowledgements
- Thanks to [DiT](https://github.com/facebookresearch/DiT) for their great work and codebase upon which we build FORA.
- Thanks to [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) for their wonderful work and contribution

## Code
Will be released **July 5**!!
