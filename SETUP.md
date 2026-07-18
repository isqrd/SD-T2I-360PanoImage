# Comprehensive Replication & Setup Guide

This guide provides step-by-step instructions to recreate the full working environment for both **SD-T2I-360PanoImage** (Stable Diffusion Image-to-360-Panorama) and **SPAG-4D** (3D Gaussian Splatting) from scratch on a new machine with an NVIDIA GPU (such as an RTX 4070Ti) running Linux or WSL2.

---

## Part 1: Stable Diffusion Image-to-360-Panorama (SD-T2I-360PanoImage)

This repository takes single or multiple input images and outpaints/inpaints them into a seamless 360° equirectangular panorama.

### 1. Environment Initialization
Create a dedicated virtual environment with Python 3.13 (or 3.10+):
```bash
cd SD-T2I-360PanoImage
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install setuptools wheel cython
```

### 2. Core Dependencies
Install the core PyTorch, Stable Diffusion, and image projection packages:
```bash
pip install torch torchvision torchaudio diffusers==0.26.0 accelerate xformers triton transformers py360convert numpy
```

### 3. Critical API and Compatibility Patches
To run under Python 3.13 with `diffusers==0.26.0` and avoid deprecated function removal errors:

1. **Downgrade `huggingface-hub`:**
   ```bash
   pip install "huggingface_hub<0.26"
   ```
2. **Downgrade `transformers`:**
   ```bash
   pip install "transformers==4.46.0"
   ```
3. **Patch & Install `basicsr` manually:**
   Modern versions of Python optimize `locals()` dictionaries which breaks older `basicsr` versions on build with a `KeyError: '__version__'`.
   ```bash
   # Download the source archive
   curl -L -o basicsr-1.4.2.tar.gz https://files.pythonhosted.org/packages/86/41/00a6b000f222f0fa4c6d9e1d6dcc9811a374cabb8abb9d408b77de39648c/basicsr-1.4.2.tar.gz
   tar -xzf basicsr-1.4.2.tar.gz
   ```
   Open `basicsr-1.4.2/setup.py` and modify the `get_version()` block around line 76 to look exactly like this (passing an explicit dictionary `g` to `exec` to avoid scope/locals optimization issues):
   ```python
   def get_version():
       g = {}
       with open(version_file, 'r') as f:
           exec(compile(f.read(), version_file, 'exec'), g)
       return g['__version__']
   ```
   Install the patched `basicsr`:
   ```bash
   pip install ./basicsr-1.4.2
   ```

4. **Patch `torchvision` functional_tensor import:**
   In newer `torchvision` versions, `transforms.functional_tensor` has been relocated.
   Open `venv/lib/python3.13/site-packages/basicsr/data/degradations.py` and modify line 8:
   * **Change:** `from torchvision.transforms.functional_tensor import rgb_to_grayscale`
   * **To:** `from torchvision.transforms.functional import rgb_to_grayscale`

5. **Install `realesrgan`:**
   ```bash
   pip install realesrgan==0.3.0
   ```

### 4. Fetch the Model Weights
Create a `models/` folder in the repository root and use the huggingface CLI to download the full suite of Stable Diffusion base, image-to-panorama, and upscaler models:
```bash
huggingface-cli download archerfmy0831/sd-t2i-360panoimage --local-dir models
```

### 5. Running the Tests
* **Single Image Outpainting:**
  ```bash
  python demo_i2p.py -i ./data/i2p-image.jpg -o result.png
  ```
* **Multi-Image (FRONT & BACK) Outpainting:**
  ```bash
  python demo_multi_i2p.py
  ```

---

## Part 2: SPAG-4D (Instant Depth-to-Splat Conversion)

This repository takes a 360° panorama and instantly projects it into a 3D Gaussian Splat.

### 1. Environment Initialization
Create a dedicated virtual environment:
```bash
cd SPAG4d
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install setuptools wheel
```

### 2. PyTorch and Torchvision Setup
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade torchvision
```

### 3. Core Requirements
```bash
pip install -r requirements.txt
```

### 4. Architecture Submodules Setup
Clone the necessary circular-disparity depth submodules:
1. **DA360 Model:**
   ```bash
   git clone https://github.com/Insta360-Research-Team/DA360 spag4d/da360_arch/DA360
   ```
2. **DAP Model:**
   ```bash
   git submodule update --init --recursive spag4d/dap_arch/DAP
   ```

### 5. Google Drive Downloader Patch (`gdown`)
Modern versions of `gdown` have deprecated and removed the `fuzzy` keyword argument in `gdown.download()`.
Open `spag4d/da360_model.py` and modify the download section around line 130 to use the folder downloader instead:
```python
        try:
            import gdown
            gdown.download_folder(
                url=DA360_CONFIG["gdrive_folder"],
                output=str(DA360_CACHE_DIR),
                quiet=False,
            )
```

### 6. Download Model Weights
Download the pre-trained weights for DAP, DA360, and PaGeR models:
```bash
python -m spag4d download-models
```

### 7. Run Conversion (OpenGL-to-OpenCV Axis Realignment)
Convert your generated panorama with `--flip-y` enabled to ensure correct camera-up orientation in standard viewers:
```bash
python -m spag4d convert /path/to/result.png result_splat_flipped.ply --stride 1 --flip-y
```
