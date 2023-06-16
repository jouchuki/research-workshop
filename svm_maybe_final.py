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

def extract_mfcc(file_path, n_mfcc=13, top_db=60):
    y, sr = librosa.load(file_path)
    #y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    #delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    extended_mfcc = np.concatenate(
    (
    mfcc,
    delta_mfcc
    ,
    delta2_mfcc
    ), axis=0)
    return extended_mfcc.T

def parse_emotion_from_filename(filename):
    return filename.split("-")[2]

def parse_gender_from_filename(filename):
    actor_id = int(filename.split("-")[-1][:2])
    return "M" if actor_id % 2 == 0 else "F"

def pad_mfcc(mfcc, max_len=300):
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    return mfcc[:max_len]

data_folder = "data"
actors = [f"Actor_{i:02d}" for i in range(1, 25)]

X, y = [], []

for actor in actors:
    actor_folder = os.path.join(data_folder, actor)
    for file_name in os.listdir(actor_folder):
        file_path = os.path.join(actor_folder, file_name)
        mfcc = extract_mfcc(file_path)
        mfcc = pad_mfcc(mfcc)
        X.append(mfcc.flatten())
        emotion = parse_emotion_from_filename(file_name)
        gender = parse_gender_from_filename(file_name)
        y.append((emotion, gender))

X, y = np.array(X), np.array(y)

# Create a DataFrame to hold the features and labels
data = pd.DataFrame(X)
data['emotion_gender'] = pd.Series(list(y)).apply(lambda x: f"{x[0]}_{x[1]}")

# Perform stratified splitting
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in sss.split(data, data['emotion_gender']):
    X_train, X_test = data.iloc[train_index].drop('emotion_gender', axis=1).values, data.iloc[test_index].drop('emotion_gender', axis=1).values
    y_train, y_test = y[train_index], y[test_index]

# Separate emotions from gender labels
y_train_emotion = y_train[:, 0]
y_test_emotion = y_test[:, 0]

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train_emotion)
y_test_encoded = encoder.transform(y_test_emotion)

clf = SVC()
clf.fit(X_train, y_train_encoded)

y_pred = clf.predict(X_test)
print(classification_report(y_test_encoded, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_test_encoded, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Emotions')
plt.ylabel('True Emotions')
plt.title('Confusion Matrix for SVM Emotion Classifier')
plt.show()
