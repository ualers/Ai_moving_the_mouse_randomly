import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Setting the number of cores for TensorFlow
num_cores = 2
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

# Generating synthetic data
n_movements = 10000000
x = np.random.normal(loc=512, scale=300, size=n_movements).clip(0, 1280)
y = np.random.normal(loc=384, scale=200, size=n_movements).clip(0, 1024)
speed = np.random.uniform(low=0.1, high=1.0, size=n_movements)
clicks = np.random.choice([0, 1], size=n_movements, p=[0.7, 0.3])
data = pd.DataFrame({'x': x, 'y': y, 'speed': speed, 'click': clicks})

# Normalizing the data
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Dividing into features and target
X = normalized_data.drop('click', axis=1)
y = normalized_data['click']

# Splitting into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the fully connected neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

# Saving the model
model.save('AIMM_V_0_0_3_FullyConnected.keras')

# Plotar curvas de perda
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.title('Curvas de Perda')
plt.legend()
plt.show()

# Plotar curvas de precisão
plt.plot(history.history['accuracy'], label='Precisão de Treinamento')
plt.plot(history.history['val_accuracy'], label='Precisão de Validação')
plt.xlabel('Épocas')
plt.ylabel('Precisão')
plt.title('Curvas de Precisão')
plt.legend()
plt.show()