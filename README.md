# 🧭 Autonomous Exploration (Active MASt3R-SLAM) 

This project integrates **MASt3R-SLAM** with **Webots** to create an **autonomous exploration system** where a robot explores an environment and reconstructs it in 3D, using learned monocular SLAM (MASt3R).

---

## ✅ Features

* Integrated with **Webots simulation** (via RGB camera and differential motors)
* Autonomous random exploration logic (left/right/forward)
* Periodic saving of:

  * Trajectory + Pose (`.txt`)
  * 3D point cloud (`.ply`)
  * Captured images (optional)
* Uses **MASt3R-SLAM** for tracking, relocalization, and map-building

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/davidwarchy/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM
```

### 2. Create Conda Environment

```bash
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
```

### 3. Install PyTorch (choose your CUDA version)

<details>
<summary>Install for CUDA 12.1 (Recommended for RTX 30-series)</summary>

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

</details>

### 4. Install Dependencies

```bash
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
```

(Optional for faster video loading):

```bash
pip install torchcodec==0.1
```

---

## 📥 Download Pretrained Checkpoints

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

---

## 🛠 Webots Setup (on WSL)

> **Note**: It's assumed you installed Webots using **Snap**, but this will be verified and updated in a future README version.

To run Webots in WSL:

```bash
sudo apt update
sudo apt install xvfb
```

### Launch Webots (Headless):

```bash
conda activate mast3r-slam
cd ~/MASt3R-SLAM/worlds
git checkout windows  # Optional, required for WSL compatibility
xvfb-run webots --stdout --stderr --batch --mode=realtime world.wbt
```

---

## 🚀 Running the SLAM System

After launching Webots (in another terminal):

```bash
conda activate mast3r-slam
python scripts/run_webots_slam.py
```

(Or whatever filename you're using for the integrated autonomous SLAM loop.)

---

## 🧪 Requirements Summary

* GPU (e.g., RTX 3070)
* WSL2 with GPU passthrough (Windows 11 or 10 + driver support)
* Webots installed (via Snap or direct download)

---

## 📌 Notes

* The robot uses basic exploration behavior (random turns and forward movement).
* The `Mode` transitions (`INIT`, `TRACKING`, `RELOC`, etc.) are managed internally.
* You can expand this by adding more intelligent exploration policies or map-based navigation.

---
