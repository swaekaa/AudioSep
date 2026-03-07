# Context-Aware Bandsplit Audio Separator

This directory contains the research implementation of a **Context-Aware Bandsplit Audio Separation Model** built on top of the AudioSep repository.

The goal of this architecture is to improve audio separation by introducing **cinematic context awareness** using:

* Audio context embeddings
* Speaker embeddings
* FiLM conditioning
* Bandsplit processing

The model adapts separation behavior depending on scene characteristics such as **dialogue presence, music intensity, and background effects**.

---

# Architecture Overview

Pipeline:

```
Waveform
   ↓
STFT
   ↓
Band Splitting
   ↓
Band Encoders
   ↓
Context Encoders
   ├ Audio Context Encoder
   └ Speaker Encoder
   ↓
Context Vector
   ↓
FiLM Conditioning
   ↓
Mask Decoder
   ↓
Separated Output
```

The context vector is used to modulate band features using **Feature-wise Linear Modulation (FiLM)**.

---

# Directory Structure

```
research/context_bandsplit
│
├── models
│   ├── bandsplit
│   │   └ bandsplit_backbone.py
│   │
│   ├── context
│   │   ├ audio_context_encoder.py
│   │   └ speaker_encoder.py
│   │
│   ├── fusion
│   │   └ film_fusion.py
│   │
│   └ separator.py
│
├── datasets
│   └ cinematic_dataset.py
│
├── training
│   └ train.py
│
├── inference
│
├── features
│
└── configs
```

---

# Hardware Requirements

Recommended GPU setup:

```
GPU: 16GB+ VRAM
RAM: 32GB
Storage: 500GB+
CUDA: 11+
```

Datasets are large and require significant storage.

---

# Environment Setup (GPU Machine)

Clone the repository:

```
git clone https://github.com/swaekaa/AudioSep.git
cd AudioSep
git checkout context-aware-bandsplit
```

Create environment:

```
pip install torch torchaudio torchvision
pip install numpy scipy librosa tqdm
```

---

# Dataset Setup

Datasets used:

* MUSDB18
* Divide and Remaster (DnR)
* VoxCeleb

Create dataset directory:

```
datasets/
│
├── musdb18
├── divide_remaster
└── voxceleb
```

---

## 1. Download MUSDB18

```
pip install musdb
```

Example structure:

```
datasets/musdb18/train/track1/
    mixture.wav
    vocals.wav
    drums.wav
    bass.wav
    other.wav
```

---

## 2. Download Divide and Remaster

Dataset:

```
Divide and Remaster Dataset
```

Place it inside:

```
datasets/divide_remaster/
```

---

## 3. Download VoxCeleb

Download VoxCeleb1 or VoxCeleb2.

Place dataset inside:

```
datasets/voxceleb/
```

Speaker embeddings will be extracted during training.

---

# BandSplit Encoding

Audio is processed using **Short-Time Fourier Transform (STFT)**.

Steps:

```
Waveform
   ↓
STFT
   ↓
Magnitude Spectrogram
   ↓
Frequency Band Splitting
   ↓
Band Encoders
```

Bands are processed independently to improve separation quality.

---

# Context Encoding

Two encoders extract contextual information.

## Audio Context Encoder

Extracts:

* energy patterns
* spectral dynamics
* background music characteristics

## Speaker Encoder

Extracts:

* speaker embeddings
* dialogue presence
* voice activity

These embeddings are combined:

```
context = [audio_embedding + speaker_embedding]
```

---

# FiLM Conditioning

Context modulates band features using:

```
output = gamma(context) * feature + beta(context)
```

This allows dynamic behavior depending on the scene.

---

# Training

Run training:

```
python research/context_bandsplit/training/train.py
```

Training pipeline:

```
Dataset
   ↓
Waveform
   ↓
Bandsplit Backbone
   ↓
Context Encoders
   ↓
FiLM Fusion
   ↓
Mask Decoder
   ↓
Loss
   ↓
Backpropagation
```

---

# Loss Functions

Current implementation:

```
L1 Loss
```

Recommended improvements:

```
SI-SNR Loss
SDR Loss
Spectrogram L1
```

---

# Inference

Inference scripts will be located in:

```
research/context_bandsplit/inference/
```

Example usage:

```
python separate_audio.py input.wav
```

---

# Future Work

Possible improvements:

* Temporal Context Transformer
* Multi-scene audio modeling
* CLAP audio embeddings
* Diffusion-based separation
* Real-time inference

---

# Notes

This research module is **independent of the core AudioSep pipeline** to avoid breaking the original implementation.

All experimentation is contained inside:

```
research/context_bandsplit/
```

---

# Author

Ekaansh Sawaria
Manipal University Jaipur
