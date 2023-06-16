import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import L2
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Input,
    TimeDistributed,
    LSTM,
    Conv1D,
    MaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# Perform FFT and extract mel spectrogram from audio file
def extract_mel_spectrogram(file_path, n_mels=128):
    y, sr = librosa.load(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)
# Parse emotion and gender from file name
def parse_emotion_and_gender_from_filename(filename):
    emotion = filename.split("-")[2]
    gender = "male" if int(filename.split("-")[-1].split(".")[0]) % 2 == 0 else "female"
    return emotion, gender

# Pad mel spectrogram to a fixed length
def pad_mel_spectrogram(mel_spectrogram, max_len=216):
    if mel_spectrogram.shape[1] < max_len:
        pad_width = max_len - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mel_spectrogram[:, :max_len]

checkpoint_path = ".\CNN_RNN_Checkpoint\cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

# Define data folder and actors
data_folder = "data"
actors = [f"Actor_{i:02d}" for i in range(1, 25)]

X, y = [], []

# Load and process data
for actor in actors:
    actor_folder = os.path.join(data_folder, actor)
    for file_name in os.listdir(actor_folder):
        file_path = os.path.join(actor_folder, file_name)
        mel_spectrogram = extract_mel_spectrogram(file_path)
        mel_spectrogram = pad_mel_spectrogram(mel_spectrogram)
        X.append(mel_spectrogram)
        emotion, _ = parse_emotion_and_gender_from_filename(file_name)
        y.append(emotion)

# Encode labels
encoder = LabelEncoder()
X, y = np.array(X), np.array(y)
data = pd.DataFrame(list(zip(X, y)), columns=['mel_spectrogram', 'emotion'])
y_encoded = encoder.fit_transform(data['emotion'])

# Split data into training and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(data, y_encoded):
    X_train, X_test = data.iloc[train_index]['mel_spectrogram'].values, data.iloc[test_index]['mel_spectrogram'].values
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

# Preprocess data
X_train, X_test = np.stack(X_train), np.stack(X_test)
X_train, X_test = np.expand_dims(X_train, -1), np.expand_dims(X_test, -1)
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

input_shape = X_train.shape[1:]

print(input_shape)

model = Sequential([
    Input(shape=input_shape),#(X_train.shape[1], X_train.shape[2])),
    Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),#Conv1D(filters=64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),#MaxPooling1D(pool_size=2),
    Dropout(rate=0.5),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),#Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),#MaxPooling1D(pool_size=2),
    Dropout(rate=0.5),
    TimeDistributed(Flatten()),
    LSTM(units=128, return_sequences=True, dropout=0.2),
    LSTM(units=64, return_sequences=False, dropout=0.2),
    Dense(units=64, activation='relu'),
    Dropout(rate=0.5),
    Dense(units=len(encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train_categorical, batch_size=64, epochs=2000, validation_data=(X_test, y_test_categorical), verbose=1, callbacks=[model_checkpoint_callback])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted Emotions")
plt.ylabel("True Emotions")
plt.title("Confusion Matrix for HCNN Emotion Classifier")
plt.show()