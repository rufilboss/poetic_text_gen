# Poetic Text Generator

The **Poetic Text Generator** is a machine learning project that uses an LSTM (Long Short-Term Memory) model to generate sequences of poetic text. This project trains on a subset of Shakespeare's works, using TensorFlow and Keras to build and train the model. The model is designed to generate text that mimics the style of poetry by predicting the next character in a sequence based on the previous characters.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Training](#model-training)
- [Generating Text](#generating-text)
- [License](#license)

## Project Overview

The **Poetic Text Generator** processes a large dataset of text (Shakespeare's work in this case) and uses this data to generate new poetic text. It does this by training a deep learning model on sequences of text and then sampling from the trained model to generate new sequences of characters that resemble the original text.

## Model Architecture

The model is built using a simple architecture consisting of:

- A single LSTM layer with 128 units.
- A dense output layer with softmax activation to output probabilities for each character.
- The model uses **categorical cross-entropy** as the loss function and **RMSprop** optimizer to improve model performance.

## Requirements

To run the **Poetic Text Generator**, ensure that you have the following packages installed:

- Python 3.x
- TensorFlow
- NumPy

The project has been tested with:

- TensorFlow 2.x
- NumPy 1.24.x

### Additional Python Packages

- TensorFlow
- NumPy

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/rufilboss/poetic-text-gen.git
   cd poetic-text-gen
   ```

2. Install the required packages using `pip`:

   ```bash
   pip install tensorflow numpy
   ```

3. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For Linux/MacOS
   venv\Scripts\activate.bat  # For Windows
   ```

## Usage

Once you have the necessary packages installed and the model trained, you can generate text using the following command:

```bash
python3 main.py
```

The output will display several generated text sequences using different "temperature" settings (0.2, 0.4, 0.6, 0.8, 1.0). Higher temperatures result in more random and creative text, while lower temperatures produce more conservative and predictable text.

## How It Works

1. **Text Preprocessing**:
   - The dataset is a collection of Shakespeare's works, which is downloaded automatically.
   - The text is tokenized, and characters are mapped to integers.

2. **Model Architecture**:
   - The model takes sequences of 40 characters as input and predicts the next character in the sequence.
   - It uses an LSTM layer for temporal sequence prediction and outputs probabilities for each character.

3. **Text Generation**:
   - The model samples the next character based on the probabilities predicted by the network.
   - Temperature controls the randomness of the predictions, where higher values increase diversity in the generated text.

## Model Training

The model is trained for 4 epochs with a batch size of 256. The dataset is processed in steps of 3 characters, generating sequences of 40 characters to train the LSTM network.

You can modify the training by changing the parameters (e.g., `batch_size`, `epochs`, `learning_rate`) in the script.

## Generating Text

Once the model is trained, it can generate new poetic text by sampling from the output predictions. The `generate_text` function handles this process, and the temperature parameter can be adjusted to control the creativity of the generated text.

You can experiment with different temperature values to produce different styles of generated poetry.

## License

This project is open-source and available under the [MIT License](LICENSE).

---
