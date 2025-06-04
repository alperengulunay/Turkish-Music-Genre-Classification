from pathlib import Path

readme_content = """
# 🎼 Turkish Music Genre Classification with Deep Learning

## 📌 Overview / Introduction

This project explores **automatic genre classification of Turkish music** using deep learning. By transforming short 2-second segments of audio into **mel-spectrogram images**, we trained a convolutional neural network (CNN) to classify songs into one of 10 Turkish music genres. This approach offers a fast, cost-effective solution for audio content categorization, which can benefit music streaming services, digital archiving, or music recommendation engines tailored to local cultural preferences.

## 🎯 Problem Statement / Context

While global music genre classification has been well-explored, Turkish music presents a unique challenge due to its **distinct rhythm patterns, scales (makams), and instrumentation**. Traditional classification systems are not optimized for this. 

The project's goal is to:
- Build a robust image-based classification pipeline for Turkish music genres.
- Evaluate the impact of various sampling rates (e.g., 11025, 22050, 44100 Hz).
- Enable quick genre detection from short music excerpts (2 seconds).
- Achieve high classification accuracy on a previously unexplored local dataset.

## 🛠️ Solution Approach & Architecture

The core idea is to convert each 2-second music segment into a mel-spectrogram image, then feed it into a **CNN model (InceptionResNetV2)** for training and prediction.

### 🎵 Workflow

1. **Data Collection**: Gather 150 mp3 samples for each of 10 genres from public sources.
2. **Preprocessing**:
   - Slice each song into 2-second clips at 60s, 70s, 80s, and 90s offsets.
   - Generate mel-spectrograms using `librosa`.
   - Save these as 224x224 images.
3. **Model Training**:
   - Fine-tune InceptionResNetV2 on the spectrogram dataset.
   - Evaluate performance with test data.
4. **Inference**:
   - Load a 2-second audio clip and classify its genre from its spectrogram.

### 🧭 Architecture Diagram

![Architecture Diagram](https://user-images.githubusercontent.com/68849018/229026800-dcef60f6-f698-42b3-a5e8-b41b8cebcd74.png)

## 📊 Data and Methods

- **Genres**: Arabesk, Caz, Elektronik, Punk, Tasavvuf, and 5 others.
- **Total Data**: 1500 songs × 4 segments = 6000 mel-spectrogram images.
- **Sampling Rates Tested**: 11025, 22050, **44100 (optimal)**.
- **Feature Extraction**: 
  - Used `librosa.feature.melspectrogram` with `n_fft=512`, `n_mels=64`, `hop_length=256`.
- **Model**: 
  - Transfer learning on **InceptionResNetV2**.
  - Input shape: (224, 224, 3).
  - Optimizer: Adam, with learning rate tuning.
  - Loss: Categorical cross-entropy.
  - Early stopping and validation monitoring applied.

## 💻 Technologies Used

| Category       | Tool/Library             | Version  |
|----------------|--------------------------|----------|
| Programming    | Python                   | 3.8+     |
| Audio Processing | librosa                 | 0.10.1   |
| Visualization  | matplotlib               | 3.7.1    |
| Deep Learning  | TensorFlow / Keras       | 2.11.0+  |
| Notebook       | Jupyter / Google Colab   | –        |
| Model          | InceptionResNetV2 (Keras.applications) | – |

## ⚙️ Installation / Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alperengulunay/Turkish-Music-Genre-Classification.git
   cd Turkish-Music-Genre-Classification
