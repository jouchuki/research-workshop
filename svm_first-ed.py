import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def extract_mfcc(file_path, n_mfcc=13, top_db=60):
    y, sr = librosa.load(file_path)
    #y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def parse_emotion_from_filename(filename):
    return filename.split("-")[2]

def pad_mfcc(mfcc, max_len=300):
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    return mfcc[:max_len]

data_folder = "data"
train_actors = [f"Actor_{i:02d}" for i in range(1, 25) if i not in (1, 2, 3, 4)]
test_actors = [f"Actor_{i:02d}" for i in range(1,5)]

X_train, X_test, y_train, y_test = [], [], [], []

for actor in train_actors:
    actor_folder = os.path.join(data_folder, actor)
    for file_name in os.listdir(actor_folder):
        file_path = os.path.join(actor_folder, file_name)
        mfcc = extract_mfcc(file_path)
        mfcc = pad_mfcc(mfcc)  # Pad MFCC sequences to have the same length
        X_train.append(mfcc.flatten())  # Flatten the MFCC features
        y_train.append(parse_emotion_from_filename(file_name))

for actor in test_actors:
    actor_folder = os.path.join(data_folder, actor)
    for file_name in os.listdir(actor_folder):
        file_path = os.path.join(actor_folder, file_name)
        mfcc = extract_mfcc(file_path)
        mfcc = pad_mfcc(mfcc)  # Pad MFCC sequences to have the same length
        X_test.append(mfcc.flatten())  # Flatten the MFCC features
        y_test.append(parse_emotion_from_filename(file_name))

X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

# Encode emotion labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Train the SVM classifier
clf = SVC()
clf.fit(X_train, y_train_encoded)

# Test the SVM classifier
y_pred = clf.predict(X_test)
print(classification_report(y_test_encoded, y_pred, target_names=encoder.classes_))

# Create a confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Emotions')
plt.ylabel('True Emotions')
plt.title('Confusion Matrix for SVM Emotion Classifier')
plt.show()