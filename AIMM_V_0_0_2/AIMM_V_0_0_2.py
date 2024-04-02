### Artificial intelligence for moving the mouse ###
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
num_cores = 4
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
n_movimentos = 10000000
x = np.random.normal(loc=512, scale=300, size=n_movimentos).clip(0, 1280)
y = np.random.normal(loc=384, scale=200, size=n_movimentos).clip(0, 1024)
velocidade = np.random.uniform(low=0.1, high=1.0, size=n_movimentos)
cliques = np.random.choice([0, 1], size=n_movimentos, p=[0.7, 0.3])
dados = pd.DataFrame({'x': x, 'y': y, 'velocidade': velocidade, 'clique': cliques})
scaler = MinMaxScaler()
dados_normalizados = pd.DataFrame(scaler.fit_transform(dados), columns=dados.columns)
X = dados_normalizados.drop('clique', axis=1)
y = dados_normalizados['clique']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=True),  
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=True),  
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=True),  
    Dropout(0.2),
    LSTM(16, activation='relu'),  
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)
model.save('AIMM_V_0_0_2.keras')
