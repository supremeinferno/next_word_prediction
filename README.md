# WordFlow

LSTM-based next word prediction system built using TensorFlow, Keras, and Streamlit.

**Live Demo:**  
https://supremeinferno-next-word-prediction.streamlit.app

---

## Overview

WordFlow is a deep learning application that predicts the most likely next word for a given text sequence.

The model was trained on a text corpus using an LSTM (Long Short-Term Memory) neural network. Given an input phrase, the application generates ranked next-word suggestions along with confidence scores.

The project demonstrates sequence modeling, text tokenization, and language generation using recurrent neural networks.

---

## Screenshots

### Application Interface

<p align="center">
  <img src="assets/home.png" width="900">
</p>

### Prediction Output

<p align="center">
  <img src="assets/prediction.png" width="900">
</p>

---

## Dataset

The model was trained on a quote-based text corpus stored in:

```text
quote_dataset.csv
```

The dataset was used to learn word sequences and contextual relationships between words for next-word prediction.

---

## Methodology

### Data Preparation

- Text normalization
- Tokenization
- Sequence generation
- Vocabulary creation
- Padding for fixed-length sequences

### Model Training

- Input sequence creation
- Word embedding representation
- LSTM-based sequence modeling
- Softmax output layer for word prediction

### Inference Pipeline

```text
Input Text
    ↓
Tokenizer
    ↓
Sequence Encoding
    ↓
Padding
    ↓
LSTM Model
    ↓
Probability Distribution
    ↓
Top-K Word Predictions
```

---

## Model Architecture

The application uses a trained LSTM language model.

Components include:

- Tokenizer
- Embedding Layer
- LSTM Layer
- Dense Output Layer
- Softmax Activation

Saved artifacts:

```text
lstm_model.h5
tokenizer.pkl
max_len.pkl
```

---

## Features

- Next word prediction
- Multiple prediction suggestions
- Confidence score visualization
- Real-time inference
- Interactive Streamlit interface
- Deep learning based language modeling

---

## Technologies Used

### Deep Learning

- TensorFlow
- Keras

### Data Processing

- NumPy
- Pickle

### Deployment

- Streamlit

---

## Project Structure

```text
next_word_prediction/
│
├── app.py
├── lstm_file.ipynb
├── quote_dataset.csv
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
├── README.md
│
└── assets/
    ├── home.png
    └── prediction.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/supremeinferno/next_word_prediction.git
```

Move into the project directory:

```bash
cd next_word_prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Sample Predictions

| Input Sequence | Predicted Word |
|----------------|----------------|
| today is a beautiful | day |
| machine learning is | powerful |
| deep learning models | can |
| the future of ai | is |

---

## Future Improvements

- Transformer-based language models
- Attention mechanisms
- Beam search decoding
- Larger training corpus
- Sentence completion generation
- Multi-word prediction

---

## Author

**Pranav Garg**

GitHub: https://github.com/supremeinferno
