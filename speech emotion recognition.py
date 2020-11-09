#!/usr/bin/env python
# coding: utf-8

import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score 


def extract_features(filename, mfcc, chroma, mel):
    with soundfile.SoundFile(filename) as sf:
        X = sf.read(dtype = "float32")
        samplerate = sf.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfcc = np.mean(librosa.feature.mfcc(y=X, sr = samplerate, n_mfcc=40).T,axis= 0)
        result= np.hstack((result,mfcc))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=samplerate).T,axis=0)
        result =np.hstack((result,mel))
    return result

emotions={"01":"neutral",
         "02":"calm",
         "03":"happy",
         "04":"sad",
         "05":"angry",
         "06":"fearful",
         "07":"disgust",
         "08":"surprised"}
emotions_here=["calm","happy","sad","disgust"]

def load(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\owner\\Documents\\ser\\Actor_*\\*.wav"):
        filename=os.path.basename(file)
        emotion=emotions[filename.split("-")[2]]
        if emotion not in emotions_here:
            continue
        feature=extract_features(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train,x_test,y_train,y_test=load(test_size=0.25)

print((x_train.shape[0], x_test.shape[0]))

print(f'Features extracted: {x_train.shape[1]}')

model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

y_pred

accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print(accuracy)

