# Model Architecture & Design

This document details the machine learning architecture and design decisions powering **deepfake_voice_recognition**.

---

## 1. Architecture Diagram

![Model Architecture](../assets/model_architecture.svg)

---

## 2. Architecture Overview

The system combines self-supervised speech representations with temporal convolutional feature aggregation:

```mermaid
flowchart TD
    %% Subgraph 1: Audio Input & Preprocessing
    subgraph AudioInput["1. Audio Input & Feature Extraction"]
        A["Raw Audio Waveform<br/>(16 kHz, Mono)"]
        B["Wav2Vec2 Feature Extractor<br/>(Zero-Mean & Unit-Variance)"]
        C["Normalized Tensor<br/>input_values (Batch, Time)"]
        A --> B
        B --> C
    end

    %% Subgraph 2: Wav2Vec Backbone
    subgraph Backbone["2. Wav2Vec 2.0 Encoder Backbone"]
        D["7-Layer CNN Feature Extractor<br/>(Frozen)"]
        E["Transformer Encoder Layers 0–7<br/>(Frozen)"]
        F["Transformer Encoder Layers 8–11<br/>(Fine-Tuned)"]
        G["Hidden Representation<br/>(Batch, Time, 768)"]
        C --> D
        D --> E
        E --> F
        F --> G
    end

    %% Subgraph 3: 1D ResNet Temporal Head
    subgraph ResNetHead["3. 1D ResNet Temporal Classification Head"]
        H["1D Input Projection<br/>(768 → 128 Channels)"]
        I["Stage 1: ResBlock1D<br/>(128 Channels, Stride 1)"]
        J["Stage 2: ResBlock1D<br/>(256 Channels, Stride 2)"]
        K["Stage 3: ResBlock1D<br/>(512 Channels, Stride 2)"]
        L["Adaptive Average Pooling 1D<br/>(512-dim Vector)"]
        M["Dense Classifier Block<br/>Dense(512→64) + BN + ReLU + Dropout(0.3)"]
        N["Linear Output Layer<br/>(64 → 2 Logits)"]
        G --> H
        H --> I
        I --> J
        J --> K
        K --> L
        L --> M
        M --> N
    end

    %% Subgraph 4: Predictions
    subgraph Output["4. Prediction Output"]
        O["Softmax Layer<br/>(Probability Distribution)"]
        P["Class 0: Genuine Voice"]
        Q["Class 1: Spoof / Deepfake Voice"]
        N --> O
        O --> P
        O --> Q
    end

    %% Styling for High Contrast and Dark/Light Theme Compatibility
    classDef inputNode fill:#0284c7,stroke:#0369a1,stroke-width:2px,color:#ffffff;
    classDef backboneNode fill:#4f46e5,stroke:#4338ca,stroke-width:2px,color:#ffffff;
    classDef resnetNode fill:#059669,stroke:#047857,stroke-width:2px,color:#ffffff;
    classDef outputNode fill:#d97706,stroke:#b45309,stroke-width:2px,color:#ffffff;
    classDef genuineNode fill:#16a34a,stroke:#15803d,stroke-width:2px,color:#ffffff;
    classDef fakeNode fill:#dc2626,stroke:#b91c1c,stroke-width:2px,color:#ffffff;

    class A,B,C inputNode;
    class D,E,F,G backboneNode;
    class H,I,J,K,L,M,N resnetNode;
    class O outputNode;
    class P genuineNode;
    class Q fakeNode;
```

---

## 3. Wav2Vec 2.0 Backbone

- **Default Pretrained Model**: `facebook/wav2vec2-base` (or `facebook/wav2vec2-large-xlsr-53` for multilingual Indic speech).
- **Self-Supervised Pretraining**: Wav2Vec 2.0 learns rich acoustic and phonetic speech representations from thousands of hours of raw speech.
- **Layer-Freezing Strategy**:
  - The convolutional raw audio encoder is **frozen** to preserve fundamental acoustic filter banks.
  - The first 8 transformer layers are **frozen** to maintain generic speech phonetic representations.
  - The top **4 transformer layers** (`cfg.num_unfrozen_layers = 4`) are **fine-tuned** to adapt speech features to synthetic artifacts (vocoder phase discontinuities, spectral unnaturalness, robotic pitch jitter).

---

## 4. 1D ResNet Temporal Classification Head

While Wav2Vec2 outputs a sequence of frame-level representations $(B, T, D)$, audio deepfake detection requires capturing localized temporal anomalies.

### Residual Block (`ResBlock1D`)
Each block applies:
1. `Conv1d(in_ch, out_ch, kernel=3, padding=1, stride=stride)`
2. `BatchNorm1d(out_ch)`
3. `ReLU(inplace=True)`
4. `Conv1d(out_ch, out_ch, kernel=3, padding=1, stride=1)`
5. `BatchNorm1d(out_ch)`
6. Skip connection with $1 \times 1$ conv downsampling when dimensions or strides change.

### Multi-Scale Downsampling
- 3 stages downsample the time dimension while expanding channel capacity ($128 \to 256 \to 512$).
- Adaptive average pooling pools all temporal frames into a fixed 512-dimensional vector.
- Dense classification head with dropout (0.3) yields calibrated class logits.

---

## 5. Separation of Preprocessing & Forward Pass

A critical engineering design decision in `model.py` is keeping `model.preprocess()` **outside** the `forward()` method:
- **Hugging Face `Wav2Vec2FeatureExtractor`**: Uses NumPy-based normalization and padding which is not directly traceable by TorchScript or ONNX exporters.
- **Traceable `forward()`**: Accepts pure PyTorch `input_values` tensors $(B, T)$ and produces $(B, 2)$ logits.
- **Result**: Enables seamless ONNX export and TorchScript compilation for low-latency production deployment.
