
### <div align="center">👉 FORA: Fast-Forward Caching in Diffusion Transformer Acceleration<div> 
<div align="center">
<a href="https://arxiv.org/abs/2407.01425"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:FORA&color=red&logo=arxiv"></a> &ensp;
</div>
Diffusion transformers (DiT) have become the de facto choice for generating high-quality images and videos, largely due to their scalability, which enables the construction of larger models for enhanced performance. However, the increased size of these models leads to higher inference costs, making them less attractive for real-time applications. We present __F__ast-F**OR**ward C**A**ching (FORA), a simple yet effective approach designed to accelerate DiT by exploiting the repetitive nature of the diffusion process. FORA implements a caching mechanism that stores and reuses intermediate outputs from the attention and MLP layers across denoising steps, thereby reducing computational overhead. This approach does not require model retraining and seamlessly integrates with existing transformer-based diffusion models. Experiments show that FORA can speed up diffusion transformers several times over while only minimally affecting performance metrics such as the IS Score and FID. By enabling faster processing with minimal trade-offs in quality, FORA represents a significant advancement in deploying diffusion transformers for real-time applications. Code will be made publicly available.

![Teaser](FORA_teaser.png)

# 🔧 Dependencies and Sampling

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13.0+cu11.7](https://pytorch.org/)
```bash
conda create -n FORA python=3.9
conda activate FORA
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```
## DiT sampling
- To sample for single ImageNet class with conditional guidance strength 1.5, with caching frequency 3, with output image size 512 and with DDIM steps 250
```bash
python src/sample.py --save-cache 'boost_infer_static' --cache-subtype 'default' --cache-threshold '3' --image-size 512 --seed 1 --cfg-scale 1.5 --num-sampling-steps 250
```
- To sample for entire ImageNetdataset and save the output in samples folder
```bash
 torchrun --nnodes=1 --nproc_per_node=4 src/sample_ddp.py --num-fid-samples 50000 --save-cache 'boost_infer_static' --cache-subtype 'default' --cache-threshold '3' --image-size 256 --per-proc-batch-size 4 --sample-dir 'samples' --cfg-scale 1.5 --num-sampling-steps 250
```

## PixelAlpha Sampling
Coming up!!
## Acknowledgements
- Thanks to [DiT](https://github.com/facebookresearch/DiT) for their great work and codebase upon which we build FORA.
- Thanks to [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) for their wonderful work and contribution

