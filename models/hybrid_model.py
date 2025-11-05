from tensorflow.keras import layers, models, optimizers

def build_hybrid(seq_len=128,channels=4):
    x_in=layers.Input(shape=(seq_len,channels))
    x=layers.Conv1D(32,5,padding='same',activation='relu')(x_in)
    x=layers.BatchNormalization()(x)
    x=layers.MaxPooling1D(2)(x)
    x=layers.Bidirectional(layers.LSTM(64))(x)
    x=layers.Dropout(0.3)(x)
    x=layers.Dense(64,activation='relu')(x)
    out=layers.Dense(1,activation='sigmoid')(x)
    m=models.Model(x_in,out)
    m.compile(optimizer=optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return m
