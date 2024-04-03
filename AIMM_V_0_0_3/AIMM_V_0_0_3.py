import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Carregar o modelo pré-treinado
model = load_model('AIMM_V_0_0_2.keras')

# Definir novas resoluções e configurações de variação de movimento
n_new_movements = 1000000
resolucoes_x = np.random.normal(loc=512, scale=300, size=n_new_movements).clip(0, 1152)
resolucoes_y = np.random.normal(loc=384, scale=200, size=n_new_movements).clip(0, 864)
new_velocities = np.random.uniform(low=0.1, high=1.0, size=n_new_movements)
new_clicks = np.random.choice([0, 1], size=n_new_movements, p=[0.7, 0.3])

# Create DataFrame with the new data
new_data = pd.DataFrame({'x': resolucoes_x, 'y': resolucoes_y, 'velocidade': new_velocities, 'click': new_clicks})

# Create a new scaler and adjust it to the new data
scaler = MinMaxScaler()
scaler.fit(new_data[['x', 'y', 'velocidade']])

# Normalize the new data
new_normalized_data = pd.DataFrame(scaler.transform(new_data[['x', 'y', 'velocidade']]), columns=['x', 'y', 'velocidade'])
new_normalized_data['click'] = new_data['click']

X_novos = new_normalized_data.drop('click', axis=1)
y_novos = new_normalized_data['click']
# Add a new dimension to the input data
X_novos_expanded = np.expand_dims(X_novos, axis=1)

# Train the model with the new data
history = model.fit(X_novos_expanded, y_novos, epochs=5, validation_split=0.2, batch_size=64)

# Salvar o modelo fine-tuned com os novos dados
model.save('AIMM_V_0_0_3.keras')
