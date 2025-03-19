<<<<<<< HEAD
# Speech Emotion Recognition

This project implements a Speech Emotion Recognition system using Python. It can classify emotions from speech audio using machine learning techniques.

## Features
- Real-time emotion recognition from microphone input
- Pre-trained model using the RAVDESS dataset
- Support for multiple emotions (happy, sad, angry, neutral, fearful, disgust, surprise)
- Audio visualization capabilities

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the program:
```bash
python emotion_recognition.py
```

## Usage
1. When you run the program, it will first train the model using the RAVDESS dataset
2. After training, you can:
   - Press 'r' to record audio from your microphone
   - Press 'q' to quit the program

## Technical Details
- Uses librosa for audio feature extraction
- Implements MFCC (Mel-frequency cepstral coefficients) features
- Uses Random Forest Classifier for emotion classification
- Supports real-time audio processing 
=======
# Speech-Emotion-Recognition
>>>>>>> 77372e79f8c9db9294c39e0ae5869d9782bc174a
