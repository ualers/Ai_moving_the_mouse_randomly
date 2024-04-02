import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('AIMM_V1.keras')

# Define new resolutions and motion variations settings
n_new_movements = 5000000
resolucoes_x = np.random.normal(loc=512, scale=300, size=n_new_movements).clip(0, 1280)
resolucoes_y = np.random.normal(loc=384, scale=200, size=n_new_movements).clip(0, 1024)
new_velocities = np.random.uniform(low=0.1, high=1.0, size=n_new_movements)
new_clicks = np.random.choice([0, 1], size=n_new_movements, p=[0.7, 0.3])

# Create DataFrame with the new data
new_data = pd.DataFrame({'x': resolucoes_x, 'y': resolucoes_y, 'velocidade': new_velocities, 'clique': new_clicks})

# Create a new scaler and adjust it to the new data
scaler = MinMaxScaler()
scaler.fit(new_data[['x', 'y', 'speed']])

# Normalize the new data
new_normalized_data = pd.DataFrame(scaler.transform(new_data[['x', 'y', 'velocidade']]), columns=['x', 'y', 'velocidade'])
new_normalized_data['click'] = new_data['click']

X_novos = new_normalized_data.drop('click', axis=1)
y_novos = new_normalized_data['click']
# Add a new dimension to the input data
X_novos_expanded = np.expand_dims(X_novos, axis=1)

# Train the model with the new data
history = model.fit(X_novos_expanded, y_novos, epochs=5, validation_split=0.2, batch_size=64)

# Save the fine-tuned model with the new data
model.save('AIMM_V1_fine_tuned_with_new_data.keras')