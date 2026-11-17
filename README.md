**Before the Ink Dries: Turning Diffusion Away from Unsafe Content In Flight**

> Text-to-image diffusion systems are widely used in creative and consumer services, yet their capacity to generate restricted content, including sexually explicit imagery, raises safety and ethical concerns. Existing defenses include system-level *filters* at the system level and two forms of pipeline-level intervention, *i.e.*, *always-on interventions* and *risk-triggered interventions*. Filters leave the pipeline's capacity for unsafe generation intact, while always-on and early interventions based on uncertain cues can disrupt benign generation. Waiting for clearer evidence, however, leaves fewer denoising steps for suppressing unsafe content. To address this tension, we propose TACK, an inference-time defense that couples visual-semantic risk assessment with latent rewind to extend the correction window while keeping the generator parameters frozen. A lightweight bridge learns to align intermediate denoiser features with a frozen vision-language teacher's final-image representations for decoding-free risk assessment via dedicated heads or an optional open-vocabulary readout. When risk is detected, TACK partially re-noises the current latent and re-denoises it under concept-excluding guidance. This additional interval provides concept suppression after late detection while reusing accumulated visual structure. Across five benchmarks, TACK achieves an average nudity erasure success rate of 99.72%, exceeding the mean performance of ten baselines by 14.73 percentage points while maintaining competitive image quality and benign prompt alignment. Across four kinds of adversarial attack strategies, TACK achieves an average attack success rate of 1.57%. With model-specific bridge training, TACK supports both U-Net- and transformer-based generators, including four Stable Diffusion variants and FLUX.1-dev.

```text
TACK/
├── model.py          # Bridge, frozen CLIP, concept head, and model loading.
├── pipeline.py       # Stand SD generation and TACK latent rewind/repair.
├── inference.ipynb   # SD1.4 inference and Original/TACK image comparison.
├── training.ipynb    # Bridge/head training, threshold calibration, and checkpoint export.
└── requirements.txt  # Python dependencies.
```
## Installation

Run the following commands from the repository root:

```bash
conda create -n tack python=3.12 pip -y
conda activate tack

python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Select **Python (TACK)** as the kernel when opening the notebooks.