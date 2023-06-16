import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def extract_chroma(file_path):
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma.T

def extract_mfcc(file_path, n_mfcc=13, top_db=10):
    y, sr = librosa.load(file_path)
    #y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    #delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    extended_mfcc = np.concatenate(
    (
    mfcc
    ,
    delta_mfcc
    #,
    #delta2_mfcc
    ), axis=0)
    return extended_mfcc.T

def parse_emotion_from_filename(filename):
    # Extract emotion and intensity
    emotion = filename.split("-")[2]
    intensity = filename.split("-")[3]
    return f"{emotion}-{intensity}"

def parse_gender_from_filename(filename):
    actor_id = int(filename.split("-")[-1][:2])
    return "M" if actor_id % 2 == 0 else "F"

def pad_features(mfcc, max_len=200):
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    return mfcc[:max_len]

def extract_melspectrogram(file_path):
    y, sr = librosa.load(file_path)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    return melspec.T

def extract_spectral_contrast(file_path):
    y, sr = librosa.load(file_path)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return spectral_contrast.T

def extract_features(file_path):
    mfcc = extract_mfcc(file_path)
    chroma = extract_chroma(file_path)
    #melspectrogram = extract_melspectrogram(file_path)
    #spectral_contrast = extract_spectral_contrast(file_path)
    combined_features = np.concatenate(
        (mfcc,
         chroma,
         #spectral_contrast,
         #melspectrogram
         ), axis=1)
    return combined_features

data_folder = "data"
actors = [f"Actor_{i:02d}" for i in range(1, 25)]

X, y = [], []

for actor in actors:
    actor_folder = os.path.join(data_folder, actor)
    for file_name in os.listdir(actor_folder):
        file_path = os.path.join(actor_folder, file_name)
        features = extract_features(file_path)
        features = pad_features(features)
        X.append(features.flatten())
        emotion_intensity = parse_emotion_from_filename(file_name)
        y.append(emotion_intensity)
X, y = np.array(X), np.array(y)

# Create a DataFrame to hold the features and labels
data = pd.DataFrame(X)
data['emotion_intensity'] = pd.Series(y)

# Perform stratified splitting
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(data, data['emotion_intensity']):
    X_train, X_test = data.iloc[train_index].drop('emotion_intensity', axis=1).values, data.iloc[test_index].drop('emotion_intensity', axis=1).values
    y_train, y_test = y[train_index], y[test_index]

print(X_train.shape)

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Normalize or standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of the explained variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train and evaluate the model with the new features and PCA
clf = SVC(kernel='rbf')
clf.fit(X_train_scaled, y_train_encoded)

y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test_encoded, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_test_encoded, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Emotions')
plt.ylabel('True Emotions')
plt.title('Confusion Matrix for SVM Emotion Classifier')
plt.show()

plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatterplot of Predicted Emotions')
plt.show()