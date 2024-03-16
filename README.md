# Artificial intelligence moving the mouse 

📚This is an artificial intelligence for moving the mouse  


initial version containing 1,000,000 movement variations

## AIMM_V1.keras

main use case:
more humanized automations


------------------basic use of the v1 model with pyautogui for movement-------------------------
        
        num_amostras = 1  
        Lista_de_movi = {
                pyautogui.easeInQuad: "Ease In Quad",
                pyautogui.easeOutQuad: "Ease Out Quad",
                pyautogui.easeInOutQuad: "Ease In Out Quad",
                pyautogui.easeInCubic: "Ease In Cubic", 
                pyautogui.easeOutCubic: "Ease Out Cubic", 
                pyautogui.easeInOutCubic: "Ease In Out Cubic", 
                pyautogui.easeInSine: "Ease In Sine", 
                pyautogui.easeOutSine: "Ease Out Sine", 
                pyautogui.easeInOutSine: "Ease In Out Sine"

        }

        diretorio_script = os.path.dirname(os.path.abspath(__file__))
        modelo_carregado = load_model(os.path.join(diretorio_script, 'AIMM_V1.keras'))
        
        x_simulado = np.random.rand(num_amostras, 1) 
        y_simulado = np.random.rand(num_amostras, 1) 
        velocidade_simulada = np.random.rand(num_amostras, 1)  
        dados_entrada_simulados = np.hstack((x_simulado, y_simulado, velocidade_simulada))
        dados_entrada_simulados = dados_entrada_simulados.reshape(num_amostras, 1, 3)
        previsao = modelo_carregado.predict(dados_entrada_simulados)
        print(previsao)
        if previsao > 0.2:
                x_tela = int(x_simulado[0][0] * pyautogui.size()[0])
                y_tela = int(y_simulado[0][0] * pyautogui.size()[1])
                rando_duracao = random.uniform(3, 4)
                funcao_aleatoria = random.choice(list(Lista_de_movi.keys()))
                frase_correspondente = Lista_de_movi[funcao_aleatoria]
                print(frase_correspondente)
                pyautogui.moveTo(x=x_tela, y=y_tela, duration=rando_duracao, tween=funcao_aleatoria)
                time.sleep(5)

## 📚Sample 
You can also choose the number of mouse movement samples requested at a time
## 📚Example
num_amostras = 2 

## 📚Customize 

You can also customize pyautogui to move and click, be creative :}
# 📚Training new model

You can also train the model with greater movement variations in AIMM.PY
        
        ## libs ###
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        # defining how many cpu to use in training
        num_cores = 4
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(num_cores)
        
        # defining the desired number of mouse movement variations
        n_movimentos = 1000000
        
        
        x = np.random.normal(loc=512, scale=300, size=n_movimentos).clip(0, 1024)
        y = np.random.normal(loc=384, scale=200, size=n_movimentos).clip(0, 768)
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
            LSTM(16, activation='relu'),  
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)
        model.save('AIMM_V1.kerass')
        
### 📚Powerful model
If you create a super powerful model of 100,000,000,000 don't hesitate to share :} 👋










                
