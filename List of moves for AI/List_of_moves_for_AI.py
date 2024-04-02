import random
import pyautogui

class movimento_com_pyautogui:
    def movimentos_do_mouse_pra_IA():
        Lista_de_movi_expandida = {
            pyautogui.linear: "Linear",
            pyautogui.easeInQuad: "Ease In Quad",
            pyautogui.easeOutQuad: "Ease Out Quad",
            pyautogui.easeInOutQuad: "Ease In Out Quad",
            pyautogui.easeInCubic: "Ease In Cubic",
            pyautogui.easeOutCubic: "Ease Out Cubic",
            pyautogui.easeInOutCubic: "Ease In Out Cubic",
            pyautogui.easeInQuart: "Ease In Quart",
            pyautogui.easeOutQuart: "Ease Out Quart",
            pyautogui.easeInOutQuart: "Ease In Out Quart",
            pyautogui.easeInQuint: "Ease In Quint",
            pyautogui.easeOutQuint: "Ease Out Quint",
            pyautogui.easeInOutQuint: "Ease In Out Quint",
            pyautogui.easeInSine: "Ease In Sine",
            pyautogui.easeOutSine: "Ease Out Sine",
            pyautogui.easeInOutSine: "Ease In Out Sine",
            pyautogui.easeInExpo: "Ease In Expo",
            pyautogui.easeOutExpo: "Ease Out Expo",
            pyautogui.easeInOutExpo: "Ease In Out Expo",
            pyautogui.easeInCirc: "Ease In Circ",
            pyautogui.easeOutCirc: "Ease Out Circ",
            pyautogui.easeInOutCirc: "Ease In Out Circ",
            pyautogui.easeInElastic: "Ease In Elastic",
            pyautogui.easeOutElastic: "Ease Out Elastic",
            pyautogui.easeInOutElastic: "Ease In Out Elastic",
            pyautogui.easeInBack: "Ease In Back",
            pyautogui.easeOutBack: "Ease Out Back",
            pyautogui.easeInOutBack: "Ease In Out Back",
            pyautogui.easeInBounce: "Ease In Bounce",
            pyautogui.easeOutBounce: "Ease Out Bounce",
            pyautogui.easeInOutBounce: "Ease In Out Bounce"
        }
        funcao_aleatoria = random.choice(list(Lista_de_movi_expandida.keys()))
        frase_correspondente = Lista_de_movi_expandida[funcao_aleatoria]
        return funcao_aleatoria, frase_correspondente