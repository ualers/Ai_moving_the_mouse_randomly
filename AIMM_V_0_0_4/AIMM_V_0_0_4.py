import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import regularizers

# Carregar o modelo pré-treinado
model = load_model('AIMM_V_0_0_3.keras')

# Definir o número total de movimentos
n_new_movements = 1000000

# Definir o tamanho do lote
batch_size = 100000

# Definir o número de lotes
num_batches = n_new_movements // batch_size

# Criar um DataFrame vazio para armazenar os dados
all_data = pd.DataFrame()

# Loop sobre os lotes
for i in range(num_batches):
    print(f'Processando lote {i+1}/{num_batches}')
    
    # Gerar dados para o lote atual
    resolucoes_x = np.random.normal(loc=512, scale=300, size=batch_size).clip(0, 1280)
    resolucoes_y = np.random.normal(loc=384, scale=200, size=batch_size).clip(0, 720)
    new_velocities = np.random.uniform(low=0.1, high=1.0, size=batch_size)
    new_clicks = np.random.choice([0, 1], size=batch_size, p=[0.7, 0.3])
    
    # Criar DataFrame com os dados do lote atual
    batch_data = pd.DataFrame({'x': resolucoes_x, 'y': resolucoes_y, 'velocidade': new_velocities, 'click': new_clicks})
    
    # Adicionar dados do lote atual ao DataFrame geral
    all_data = pd.concat([all_data, batch_data], ignore_index=True)

# Create a new scaler and adjust it to the new data
scaler = MinMaxScaler()
scaler.fit(all_data[['x', 'y', 'velocidade']])

# Normalize the new data
new_normalized_data = pd.DataFrame(scaler.transform(all_data[['x', 'y', 'velocidade']]), columns=['x', 'y', 'velocidade'])
new_normalized_data['click'] = all_data['click']

X_novos = new_normalized_data.drop('click', axis=1)
y_novos = new_normalized_data['click']

# Add a new dimension to the input data
X_novos_expanded = np.expand_dims(X_novos, axis=1)

# Criar um novo modelo
new_model = tf.keras.Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_novos_expanded.shape[1], X_novos_expanded.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=True),  
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=True),  
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=True),  
    Dropout(0.2),
    LSTM(16, activation='relu'),  
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.2)
])

# Compilar o novo modelo com uma taxa de aprendizado menor
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
new_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o novo modelo com os novos dados
history = new_model.fit(X_novos_expanded, y_novos, epochs=10, validation_split=0.5, batch_size=64)

# Salvar o modelo fine-tuned com os novos dados
new_model.save('AIMM_V_0_0_4.keras')
