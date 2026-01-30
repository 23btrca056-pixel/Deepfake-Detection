import tensorflow as tf
from tensorflow.keras import layers, models

def build_deepfake_model():

    model = models.Sequential()

    # 1. CNN Feature Extractor
    model.add(layers.TimeDistributed(
        layers.Conv2D(32, (3,3), activation='relu'),
        input_shape=(20, 128, 128, 3)
    ))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))

    model.add(layers.TimeDistributed(
        layers.Conv2D(64, (3,3), activation='relu')
    ))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))

    model.add(layers.TimeDistributed(layers.Flatten()))

    # 2. LSTM / BiLSTM for Temporal Detection
    model.add(layers.Bidirectional(layers.LSTM(64)))

    # 3. Dense classification
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
