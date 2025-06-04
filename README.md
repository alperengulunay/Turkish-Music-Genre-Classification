# 🎼 Turkish Music Genre Classification with Deep Learning

## Introduction

This project explores **automatic genre classification of Turkish music** using deep learning. By transforming short 2-second segments of audio into **mel-spectrogram images**, we trained a convolutional neural network (CNN) to classify songs into one of 10 Turkish music genres. This approach offers a fast, cost-effective solution for audio content categorization, which can benefit music streaming services, digital archiving, or music recommendation engines tailored to local cultural preferences.

## Context

While global music genre classification has been well-explored, Turkish music presents a unique challenge due to its **distinct rhythm patterns, scales (makams), and instrumentation**. Traditional classification systems are not optimized for this. 

The project's goal is to:
- Build a robust image-based classification pipeline for Turkish music genres.
- Evaluate the impact of various sampling rates (e.g., 11025, 22050, 44100 Hz).
- Enable quick genre detection from short music excerpts (2 seconds).
- Achieve high classification accuracy on a previously unexplored local dataset.

## Solution Approach & Architecture

The core idea is to convert each 2-second music segment into a mel-spectrogram image, then feed it into a **CNN model (InceptionResNetV2)** for training and prediction.

### Workflow

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

### Architecture Diagram

![Architecture Diagram](https://github.com/user-attachments/assets/1be45d5b-ae08-4db3-9a36-a0791962c4cd)

## Data and Methods

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
   ```

2. **Create and activate virtual environment** (optional):
   ```bash
   python -m venv env
   source env/bin/activate  # or `env\Scripts\activate` on Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run preprocessing script**:
   ```bash
   python get_mel_spec_44100_using_librosa.py
   ```

5. **Train or test model** using:
   - `InceptionResNetV2.ipynb` for training.
   - `test_model.ipynb` for evaluation.

## Usage Examples

```python
# Generate a mel-spectrogram from a new audio file
from get_mel_spec_44100_using_librosa import generate_melspec
melspec = generate_melspec("example.mp3")

# Predict genre
model = load_model("your_model.h5")
prediction = model.predict(melspec)
print("Predicted genre:", decode_prediction(prediction))
```

📸 Example Spectrogram:

![Example](https://user-images.githubusercontent.com/68849018/229026822-96ae425f-8438-4874-bbd4-09768fd12098.png)

## 📈 Results and Performance

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | ~89%      |
| Precision    | Varied by genre (avg. ~88%) |
| Inference Time | < 1s per clip |

📊 Confusion Matrix & Accuracy Plot:

![Results](https://github.com/user-attachments/assets/a2ba4f4b-3331-41ec-9166-4d1d21671464)

## 💡 Business Impact / Outcome


- Enables **real-time genre classification** in Turkish music apps.
- Reduces **manual labeling** and enhances personalization for local listeners.
- Provides a foundation for **cultural music analytics** in underserved language domains.
- Scalable to integrate with recommendation engines or archiving systems.

## 🔮 Future Work

- Expand dataset with **more genres and live recordings**.
- Evaluate performance on **noisy or low-quality audio**.
- Explore **self-supervised audio pretraining** models (e.g., wav2vec).
- Deploy model via **FastAPI or Flask** for real-time REST inference.
