# 🎙 Uyghur Speech-to-Text – Whisper Fine-Tuning Project

> Task: build a high-performing ASR system capable of accurately transcribing audio clips in the Uyghur language as 2 day timed kaggle competetion.

## 📌 Overview

This project presents a complete pipeline for solving a Uyghur Automatic Speech Recognition (ASR) challenge using OpenAI Whisper-based models. The solution progresses from a baseline model to language-adapted inference and finally domain-specific fine-tuning.

The goal was to minimize Character Error Rate (CER) while maintaining efficient inference.

---

## 🏗 Project Structure

The notebook implements three major stages:

1. Baseline Whisper inference
2. Uyghur fine-tuned model inference
3. Further fine-tuning on competition dataset

---

## 🚀 Approach

### 1️⃣ Baseline Model

* Model: `whisper-small`
* Direct transcription on test set
* No language adaptation
* Used as performance reference

---

### 2️⃣ Language-Adapted Model

* Model: `ixxan/whisper-small-uyghur-thugy20`
* Beam search decoding
* Controlled temperature
* Improved recognition of Uyghur phonetics

Decoding configuration:

* `num_beams = 3`
* `temperature = 0.3`
* Deterministic generation (no sampling)

---

### 3️⃣ Domain Fine-Tuning

Fine-tuned the Uyghur-adapted checkpoint on the competition dataset.

#### Dataset Processing

* Loaded training CSV
* Constructed absolute audio paths
* Converted to HuggingFace Dataset
* Cast audio column with 16kHz resampling
* Shuffled with fixed seed

Training subset used:

* 6000 samples

---

### 🧠 Training Details

Model: WhisperForConditionalGeneration
Processor: WhisperProcessor

Training Configuration:

* Batch size: 4
* Gradient accumulation: 2
* Learning rate: 1e-5
* Epochs: 3
* Max steps: 1000
* Mixed precision enabled (fp16 if CUDA available)

Framework:

* HuggingFace Seq2SeqTrainer

---

## 🔎 Text Normalization Strategy

To reduce CER, the following post-processing was applied:

* Unicode normalization (NFKD)
* Diacritics removal
* Punctuation removal
* Whitespace normalization

This step significantly improves leaderboard scores in character-based evaluation settings.

---

## 📂 Inference Pipeline

1. Load model and processor
2. Load audio using torchaudio
3. Resample to 16kHz if required
4. Extract input features
5. Generate transcription (beam search)
6. Normalize output
7. Save submission CSV

---

## 🛠 Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* HuggingFace Datasets
* Torchaudio
* JiWER (for CER)
* Evaluate

---

## 📊 Model Evolution Strategy

| Stage | Model                     | Purpose             |
| ----- | ------------------------- | ------------------- |
| 1     | Whisper Small             | Baseline            |
| 2     | Uyghur Whisper Small      | Language adaptation |
| 3     | Fine-tuned Uyghur Whisper | Domain adaptation   |

---

## 🧩 Key Insights

* Language adaptation dramatically improves low-resource ASR.
* Beam search stabilizes predictions for morphologically rich languages.
* Text normalization meaningfully reduces CER.
* Domain fine-tuning further improves phoneme-to-grapheme mapping accuracy.

---

## ⚙️ How to Run

### Install Dependencies

```bash
pip install transformers datasets torchaudio evaluate jiwer
```

### Train

Run fine-tuning section of notebook or convert to script:

```bash
python train.py
```

### Inference

```bash
python inference.py
```

---

## 📈 Future Improvements

* Add validation split with CER tracking
* Use larger Whisper checkpoint
* Try SpecAugment during training
* Tune beam width vs speed trade-off
* Apply language model re-ranking
* Perform error analysis on frequent character substitutions

---
