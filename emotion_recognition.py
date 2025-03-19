import os
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time

class EmotionRecognizer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.sample_rate = 22050
        self.duration = 3  # seconds
        self.emotions = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
    def extract_features(self, audio_path):
        """Extract MFCC features from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Calculate statistics
            features = []
            for i in range(mfcc.shape[0]):
                features.extend([
                    np.mean(mfcc[i]),
                    np.std(mfcc[i]),
                    np.max(mfcc[i]),
                    np.min(mfcc[i])
                ])
            
            return np.array(features)
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def record_audio(self):
        """Record audio from microphone"""
        print("Recording...")
        recording = sd.rec(int(self.duration * self.sample_rate),
                         samplerate=self.sample_rate,
                         channels=1)
        sd.wait()
        print("Recording finished")
        
        # Convert to mono if stereo
        if len(recording.shape) > 1:
            recording = recording.mean(axis=1)
        
        return recording

    def process_audio(self, audio_data):
        """Process recorded audio and extract features"""
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        
        # Calculate statistics
        features = []
        for i in range(mfcc.shape[0]):
            features.extend([
                np.mean(mfcc[i]),
                np.std(mfcc[i]),
                np.max(mfcc[i]),
                np.min(mfcc[i])
            ])
        
        return np.array(features)

    def train_model(self):
        """Train the model using RAVDESS dataset"""
        print("Training model...")
        # This is a placeholder for the actual training code
        # In a real implementation, you would load the RAVDESS dataset
        # and train the model with it
        print("Model training completed")

    def predict_emotion(self, features):
        """Predict emotion from features"""
        if features is None:
            return "Error processing audio"
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict emotion
        prediction = self.model.predict(features_scaled)
        return self.emotions[prediction[0]]

    def visualize_audio(self, audio_data):
        """Visualize the audio waveform"""
        plt.figure(figsize=(10, 4))
        plt.plot(audio_data)
        plt.title('Audio Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.show()

def main():
    recognizer = EmotionRecognizer()
    
    # Train the model
    recognizer.train_model()
    
    print("\nEmotion Recognition System")
    print("Press 'r' to record audio")
    print("Press 'q' to quit")
    
    while True:
        command = input("\nEnter command: ").lower()
        
        if command == 'q':
            break
        elif command == 'r':
            # Record audio
            audio_data = recognizer.record_audio()
            
            # Visualize audio
            recognizer.visualize_audio(audio_data)
            
            # Process audio and predict emotion
            features = recognizer.process_audio(audio_data)
            emotion = recognizer.predict_emotion(features)
            
            print(f"Predicted emotion: {emotion}")
        else:
            print("Invalid command. Please try again.")

if __name__ == "__main__":
    main() 