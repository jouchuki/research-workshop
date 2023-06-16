import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = "CNN_Checkpoint/cp-1969.ckpt" #\
                  #"{epoch:04d}.ckpt"

# Load the latest checkpoint
latest_checkpoint = tf.train.latest_checkpoint("CNN_Checkpoint")
print(f"Latest checkpoint: {latest_checkpoint}")

def extract_mel_spectrogram(file_path, n_mels=128):
    y, sr = librosa.load(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

def parse_emotion_from_filename(filename):
    return filename.split("-")[2]

def pad_mel_spectrogram(mel_spectrogram, max_len=216):
    if mel_spectrogram.shape[1] < max_len:
        pad_width = max_len - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mel_spectrogram[:, :max_len]

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training and validation accuracy values
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training and validation loss values
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Validation'], loc='upper left')
    plt.show()


data_folder = "data"
actors = [f"Actor_{i:02d}" for i in range(1, 25)]

X, y = [], []

for actor in actors:
    actor_folder = os.path.join(data_folder, actor)
    for file_name in os.listdir(actor_folder):
        file_path = os.path.join(actor_folder, file_name)
        mel_spectrogram = extract_mel_spectrogram(file_path)
        mel_spectrogram = pad_mel_spectrogram(mel_spectrogram)
        X.append(mel_spectrogram)
        y.append(parse_emotion_from_filename(file_name))

encoder = LabelEncoder()

X, y = np.array(X), np.array(y)

data = pd.DataFrame(list(zip(X, y)), columns=['mel_spectrogram', 'emotion'])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

y_encoded = encoder.fit_transform(data['emotion'])
for train_index, test_index in sss.split(data, y_encoded):
    X_train, X_test = data.iloc[train_index]['mel_spectrogram'].values, data.iloc[test_index]['mel_spectrogram'].values
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

X_train, X_test = np.stack(X_train), np.stack(X_test)
X_train, X_test = np.expand_dims(X_train, -1), np.expand_dims(X_test, -1)

y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

input_shape = X_train.shape[1:]

# Create a new model instance with the same architecture as before
loaded_model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.14554514863546503),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.14554514863546503),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.14554514863546503),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=L2(0.04101629484574286)),
    Dropout(0.46643399942391695),
    #Dense(128, activation='relu', kernel_regularizer=L2(0.001)),
    #Dropout(0.5),
    Dense(len(encoder.classes_), activation='softmax'),
])

# Load the weights from the checkpoint
loaded_model.load_weights(latest_checkpoint)

# Compile the loaded model
loaded_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Feed new data to the loaded model
predictions = loaded_model.predict(X_test)

y_pred = np.argmax(loaded_model.predict(X_test), axis=-1)

print(classification_report(y_test, y_pred, target_names=encoder.classes_))
# Assuming y_test is the ground truth labels for the test set
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Emotions')
plt.ylabel('True Emotions')
plt.title('Confusion Matrix for CNN Emotion Classifier')
plt.show()

plot_history(history)