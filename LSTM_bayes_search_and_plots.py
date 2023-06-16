import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Input,
    LocallyConnected2D,
    ELU,
    LSTM,
    TimeDistributed, LSTM, LocallyConnected1D, MaxPooling1D, Reshape
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Ftrl, Nadam, AdamW
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def extract_mel_spectrogram(file_path, n_mels=128):
    y, sr = librosa.load(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

def parse_emotion_and_gender_from_filename(filename):
    emotion = filename.split("-")[2]
    gender = "male" if int(filename.split("-")[-1].split(".")[0]) % 2 == 0 else "female"
    return emotion, gender

def pad_mel_spectrogram(mel_spectrogram, max_len=251):
    if mel_spectrogram.shape[1] < max_len:
        pad_width = max_len - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mel_spectrogram[:, :max_len]

data_folder = "data"
actors = [f"Actor_{i:02d}" for i in range(1, 25)]

X, y, gender = [], [], []

for actor in actors:
    actor_folder = os.path.join(data_folder, actor)
    for file_name in os.listdir(actor_folder):
        file_path = os.path.join(actor_folder, file_name)
        mel_spectrogram = extract_mel_spectrogram(file_path)
        mel_spectrogram = pad_mel_spectrogram(mel_spectrogram)
        X.append(mel_spectrogram)
        emotion, actor_gender = parse_emotion_and_gender_from_filename(file_name)
        y.append(emotion)
        gender.append(actor_gender)

encoder = LabelEncoder()
X, y = np.array(X), np.array(y)
data = pd.DataFrame(list(zip(X, y, gender)), columns=['mel_spectrogram', 'emotion', 'gender'])
y_encoded = encoder.fit_transform(data['emotion'])

X = np.stack(data['mel_spectrogram'].values)
X = np.expand_dims(X, -1)

input_shape = X.shape[1:]

def lstm_cnn(learning_rate=0.001, optimizer='Adam', alpha=1.0):
    with tf.device('/GPU:0'):
    # Your model code here
        model = Sequential([
            Input(shape=input_shape),
            LocallyConnected2D(8, kernel_size=(3, 3), strides=(1, 1)),
            BatchNormalization(),
            ELU(alpha=alpha),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            LocallyConnected2D(8, kernel_size=(3, 3), strides=(1, 1)),
            BatchNormalization(),
            ELU(alpha=alpha),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            LocallyConnected2D(16, kernel_size=(3, 3), strides=(1, 1)),
            BatchNormalization(),
            ELU(alpha=alpha),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            LocallyConnected2D(16, kernel_size=(3, 3), strides=(1, 1)),
            BatchNormalization(),
            ELU(alpha=alpha),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            TimeDistributed(Flatten()),
            LSTM(256),
            Dense(len(encoder.classes_), activation='softmax'),
        ])

    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Adagrad':
        opt = Adagrad(learning_rate=learning_rate)
    elif optimizer == 'Adadelta':
        opt = Adadelta(learning_rate=learning_rate)
    elif optimizer == 'Ftrl':
        opt = Ftrl(learning_rate=learning_rate)
    elif optimizer == 'Nadam':
        opt = Nadam(learning_rate=learning_rate)
    elif optimizer == 'AdamW':
        opt = AdamW(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=lstm_cnn, epochs=50, batch_size=32, verbose=0)

search_spaces = {
    'learning_rate': Real(1e-6, 1e-3, prior='log-uniform'),
    'optimizer': Categorical(['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Ftrl', 'Nadam', 'AdamW']),
    'alpha': Real(0.1, 1.0),
}

bayes_search = BayesSearchCV(
    model,
    search_spaces,
    n_iter=20,
    cv=StratifiedKFold(n_splits=6, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1,
    scoring='accuracy',
    random_state=42
)

bayes_search.fit(X, y_encoded, groups=data['gender'].values)

print(f"Best score: {bayes_search.best_score_}")
print(f"Best parameters: {bayes_search.best_params_}")

# Mean accuracy plot for all models
mean_test_scores = bayes_search.cv_results_['mean_test_score']
plt.bar(range(len(mean_test_scores)), mean_test_scores)
plt.xlabel('Models')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy for Each Model')
plt.show()

# Train and evaluate the best model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
best_model = lstm_cnn(**bayes_search.best_params_)
history = best_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test))
y_pred = np.argmax(best_model.predict(X_test), axis=-1)

# Accuracy plot for the best model
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Best Model Accuracy') #(Parameters: {bayes_search.best_params_})')
plt.legend()
plt.show()

# Confusion matrix for the best model
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix') #(Best Model with Parameters: {bayes_search.best_params_})')
plt.show()