# ARTIFICIAL_INTELIGENT_MOVIMENT_MOUSE

This is an artificial intelligence for moving the mouse  


initial version containing 1,000,000 movement variations

main use case:
more humanized automations

        
        
        
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
                rando_tween = random.randint(2, 3)
                pyautogui.moveTo(x=x_tela, y=y_tela, duration=rando_duracao, tween=funcao_aleatoria)
                time.sleep(5)
