from typing import List

import numpy
import matplotlib.pyplot as plt

from tsk import estimador, gaussiana, equacao_parametrica, valor_estimado


def prepare_x(start: float, end: float, step: float) -> List[float]:
    """

    Args:
        start:
        end:
        step:

    Returns:
    """
    x: List[float] = []
    value = start
    while True:
        if value > end:
            break
        x.append(round(value, 2))
        value += step
    return x


def mount_parable(x: List[float], x_1: float, sigma_1: float, x_2: float, sigma_2: float, p_1: float, q_1: float,
                  p_2: float, q_2: float) -> List[float]:
    """Monta a parábola dado os parâmetros ajustados.

    Args:
        x: Lista dos pontos x.
        x_1: Média da primeira gaussiana.
        sigma_1: Desvio padrão da primeira gaussiana.
        x_2: Média da segunda gaussiana.
        sigma_2: Desvio padrão da segunda gaussiana.
        p_1: Primeiro parâmetro da equação paramétrica.
        q_1: Primeira constante da equação paramétrica
        p_2: Segundo parâmetro da equação paramétrica
        q_2: Segunda constante da equação paramétrica

    Returns: Lista de valores estimados.
    """
    y_d: List[float] = []
    for ponto in x:
        w_1 = gaussiana(ponto, x_1, sigma_1)
        w_2 = gaussiana(ponto, x_2, sigma_2)
        y_1 = equacao_parametrica(ponto, p_1, q_1)
        y_2 = equacao_parametrica(ponto, p_2, q_2)
        y_d.append(valor_estimado(w_1, w_2, y_1, y_2))
    return y_d


def main():
    """Execução do algoritmo.
    """
    # executando o algoritmo
    x = prepare_x(-5, 5, 0.1)
    alpha = 0.003
    epocas = 1000
    x_1, sigma_1, x_2, sigma_2, p_1, q_1, p_2, q_2, erros = estimador(x, alpha, epocas)
    # exibindo o erro médio
    epocas = prepare_x(1, epocas, 1)
    plt.plot(epocas, erros)
    plt.title('Erro médio por década')
    plt.show()
    # exibindo a função modelada
    y_d = mount_parable(x, x_1, sigma_1, x_2, sigma_2, p_1, q_1, p_2, q_2)
    plt.plot(x, y_d)
    plt.title('Parábola estimada')
    plt.show()


main()
