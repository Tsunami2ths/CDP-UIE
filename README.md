# CDP-UIE: Cross-Domain Progressive Underwater Image Enhancement

> **Official PyTorch Implementation for the paper "CDP-UIE: Cross-Domain Progressive Underwater Image Enhancement"** > 
> *The code will be fully open-sourced upon acceptance.*

![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
![PyTorch 2.0.0](https://img.shields.io/badge/pytorch-2.0.0-orange.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Introduction
Underwater image enhancement suffers from severe color distortion (due to wavelength-dependent absorption) and structural blurring (due to forward scattering). Existing single-domain methods often face feature coupling and optimization conflicts. 

To completely release the fitting potential of the dual-stage architecture, we propose **CDP-UIE**, a progressive enhancement algorithm based on **frequency-band divide-and-conquer** and **cross-domain task alignment**:
- **Stage 1: Global Refinement (GR)**. Operates in the LAB perceptual domain. Driven by low-frequency style suppression, it eliminates dominant water style and reconstructs a natural color base.
- **Stage 2: Local Detail Reconstruction (LDR)**. Returns to the physical RGB spectral domain. Guided by high-frequency directional sharpening and the Physical Spectral-Aware Decoupling Module (PSADM), it restores fragile textures and fine details.

*(Insert your network architecture diagram here: `![Network Architecture](assets/network_arch.png)`)*

---

## ⚙️ Dependencies & Environment

- **OS:** Ubuntu 20.04 or above / Windows 10
- **Python:** 3.8+
- **PyTorch:** 2.0.0+ (CUDA 11.8 recommended)

```bash
# Clone the repository
git clone [https://github.com/Tsunami2ths/CDP-UIE.git](https://github.com/YourUsername/CDP-UIE.git)
cd CDP-UIE

# Install dependencies
pip install -r requirements.txt
pip install torch==2.0.0 torchvision==0.15.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)