# Arquivo contento a lógica da modelagem do método TSK para aproximação de uma parábola (x^2) por duas gaussianas

# variáveis inicializadas aleatoriamente
import random
from math import exp
from typing import List, Tuple

X_1 = -2  # média da primeira gaussiana
SIGMA_1 = 0.2  # desvio padrão da primeira gaussiana
X_2 = 2  # média da segunda gaussiana
SIGMA_2 = 0.2  # desvio padrão da segunda gaussiana
P_1 = -2  # coeficiente da primeira equação paramétrica
Q_1 = 0  # constante da primeira equação paramétrica
P_2 = 2  # coeficiente da segunda equação paramétrica
Q_2 = 0  # constante da segunda equação paramétrica


# equação paramétrica geral
def equacao_parametrica(x: float, p: float, r: float) -> float:
    """Equação paramétrica geral para utilização no algoritmo.

    Args:
        x: Valor de x definido.
        p: Parâmetro da função paramétrica.
        r: Constante da função paramétrica.

    Returns: Valor da equação paramétrica.
    """
    return (p * x) + r


def gaussiana(x: float, media: float, desvio_p: float) -> float:
    """Gaussiana geral.

    Args:
        x: Valor de x definido.
        media: Média.
        desvio_p: Desvio padrão.

    Returns: Valor de w que representa o valor da função de pertinência.
    """
    base = (x - media) / desvio_p
    return exp(-0.5 * pow(base, 2))


def valor_estimado(w_1, w_2, y_1, y_2) -> float:
    """Função para definir o valor estimado.

    Args:
        w_1: Valor da primeira função de pertinência.
        w_2: Valor da segunda função de pertinência.
        y_1: Valor da primeira função paramétrica.
        y_2: Valor da primeira função paramétrica.

    Returns: Valor estimado.
    """
    return ((w_1 * y_1) + (w_2 * y_2)) / (w_1 + w_2)


def ajuste_erro_p_1(p_1_k: float, alpha: float, y: float, y_d: float, w_1: float, w_2: float, x: float) -> float:
    """Ajuste do erro de p_1.

    Args:
        p_1_k: P_1 da iteração atual.
        alpha: Fator de correção do erro.
        y: Valor estimado.
        y_d: Valor desejado.
        w_1: Valor da equação de pertinência.
        w_2: Valor da equação de pertinência.
        x: Valor de x.

    Returns: Valor de p1 para a próxima iteração.
    """
    derivada_p_1 = (y - y_d) * (w_1/(w_1 + w_2)) * x
    return p_1_k - (alpha * derivada_p_1)


def ajuste_erro_p_2(p_2_k: float, alpha: float, y: float, y_d: float, w_1: float, w_2: float, x: float) -> float:
    """Ajuste do erro de p_2.

    Args:
        p_2_k: P_2 da iteração atual.
        alpha: Fator de correção do erro.
        y: Valor estimado.
        y_d: Valor desejado.
        w_1: Valor da equação de pertinência.
        w_2: Valor da equação de pertinência.
        x: Valor de x.

    Returns: Valor de p2 para a próxima iteração.
    """
    derivada_p_2 = (y - y_d) * (w_2 / (w_1 + w_2)) * x
    return p_2_k - (alpha * derivada_p_2)


def ajuste_erro_q_1(q_1_k: float, alpha: float, y: float, y_d: float, w_1: float, w_2: float) -> float:
    """Ajuste do erro de q_1.

    Args:
        q_1_k: q_1 da iteração atual.
        alpha: Fator de correção do erro.
        y: Valor estimado.
        y_d: Valor desejado.
        w_1: Valor da equação de pertinência.
        w_2: Valor da equação de pertinência.

    Returns: Valor de q1 para a próxima iteração.
    """
    derivada_q_1 = (y - y_d) * (w_1 / (w_1 + w_2))
    return q_1_k - (alpha * derivada_q_1)


def ajuste_erro_q_2(q_2_k: float, alpha: float, y: float, y_d: float, w_1: float, w_2: float) -> float:
    """Ajuste do erro de p_1.

    Args:
        q_2_k: q_2 da iteração atual.
        alpha: Fator de correção do erro.
        y: Valor estimado.
        y_d: Valor desejado.
        w_1: Valor da equação de pertinência.
        w_2: Valor da equação de pertinência.

    Returns: Valor de p1 para a próxima iteração.
    """
    derivada_q_2 = (y - y_d) * (w_2 / (w_1 + w_2))
    return q_2_k - (alpha * derivada_q_2)


def ajuste_erro_x_1(x_1_k: float, alpha: float, y: float, y_d: float, w_1: float, w_2: float, x: float,
                    y_1: float, y_2: float, x_1: float, sigma_1: float) -> float:
    """Ajuste do erro de x_1.

    Args:
        x_1_k: x_1 da iteração atual.
        alpha: Fator de correção do erro.
        y: Valor estimado.
        y_d: Valor desejado.
        w_1: Valor da equação de pertinência.
        w_2: Valor da equação de pertinência.
        x: Valor de x.
        y_1: Valor de y_1.
        y_2: Valor de y_2.
        x_1: Média da primeira gaussiana.
        sigma_1: Desvio padrão da primeira gaussiana.

    Returns: Valor de p1 para a próxima iteração.
    """
    primeiro_termo = y - y_d
    segundo_termo = (y_1 - y_2) / pow(w_1 + w_2, 2)
    terceiro_termo = (x - x_1) / pow(sigma_1, 2)
    derivada_x_1 = primeiro_termo * w_2 * segundo_termo * w_1 * terceiro_termo
    return x_1_k - (alpha * derivada_x_1)


def ajuste_erro_x_2(x_2_k: float, alpha: float, y: float, y_d: float, w_1: float, w_2: float, x: float,
                    y_1: float, y_2: float, x_2: float, sigma_2: float) -> float:
    """Ajuste do erro de x_2.

    Args:
        x_2_k: x_2 da iteração atual.
        alpha: Fator de correção do erro.
        y: Valor estimado.
        y_d: Valor desejado.
        w_1: Valor da equação de pertinência.
        w_2: Valor da equação de pertinência.
        x: Valor de x.
        y_1: Valor de y_1.
        y_2: Valor de y_2.
        x_2: Média da primeira gaussiana.
        sigma_2: Desvio padrão da primeira gaussiana.

    Returns: Valor de p1 para a próxima iteração.
    """
    primeiro_termo = y - y_d
    segundo_termo = (y_2 - y_1) / pow(w_1 + w_2, 2)
    terceiro_termo = (x - x_2) / pow(sigma_2, 2)
    derivada_x_2 = primeiro_termo * w_1 * segundo_termo * w_2 * terceiro_termo
    return x_2_k - (alpha * derivada_x_2)


def ajuste_erro_sigma_1(sigma_1_k: float, alpha: float, y: float, y_d: float, w_1: float, w_2: float, x: float,
                        y_1: float, y_2: float, x_1: float, sigma_1: float) -> float:
    """Ajuste do erro de sigma_1.

    Args:
        sigma_1_k: sigma_1 da iteração atual.
        alpha: Fator de correção do erro.
        y: Valor estimado.
        y_d: Valor desejado.
        w_1: Valor da equação de pertinência.
        w_2: Valor da equação de pertinência.
        x: Valor de x.
        y_1: Valor de y_1.
        y_2: Valor de y_2.
        x_1: Média da primeira gaussiana.
        sigma_1: Desvio padrão da primeira gaussiana.

    Returns: Valor de p1 para a próxima iteração.
    """
    primeiro_termo = y - y_d
    segundo_termo = (y_1 - y_2) / pow(w_1 + w_2, 2)
    terceiro_termo = pow(x - x_1, 2) / pow(sigma_1, 3)
    derivada_sigma_1 = primeiro_termo * w_2 * segundo_termo * w_1 * terceiro_termo
    return sigma_1_k - (alpha * derivada_sigma_1)


def ajuste_erro_sigma_2(sigma_2_k: float, alpha: float, y: float, y_d: float, w_1: float, w_2: float, x: float,
                        y_1: float, y_2: float, x_2: float, sigma_2: float) -> float:
    """Ajuste do erro de sigma_2.

    Args:
        sigma_2_k: sigma_2 da iteração atual.
        alpha: Fator de correção do erro.
        y: Valor estimado.
        y_d: Valor desejado.
        w_1: Valor da equação de pertinência.
        w_2: Valor da equação de pertinência.
        x: Valor de x.
        y_1: Valor de y_1.
        y_2: Valor de y_2.
        x_2: Média da primeira gaussiana.
        sigma_2: Desvio padrão da primeira gaussiana.

    Returns: Valor de p1 para a próxima iteração.
    """
    primeiro_termo = y - y_d
    segundo_termo = (y_2 - y_1) / pow(w_1 + w_2, 2)
    terceiro_termo = pow(x - x_2, 2) / pow(sigma_2, 3)
    derivada_sigma_2 = primeiro_termo * w_2 * segundo_termo * w_1 * terceiro_termo
    return sigma_2_k - (alpha * derivada_sigma_2)


def erro(y: float, y_d: float) -> float:
    """Erro quadrático.

    Args:
        y: Y estimado.
        y_d: Y desejado.

    Returns: Erro quadrático.
    """
    return 0.5 * pow(y - y_d, 2)


def estimador(x: List[float], alpha: float, quant_epocas: int) -> Tuple[float, float, float, float, float, float, float,
                                                                        float, List[float]]:
    """ Algoritmo estimador.

    Args:
        x: Lista dos valores de x.
        alpha: Fator de correção.
        quant_epocas: Quantidade de épocas.

    Returns: x_1, sigma_1, x_2, sigma_2, p_1, q_1, p_2, q_2, Lista de erros por época.
    """
    # erros
    erros: List[float] = []
    # inicializando os parâmetro aleatoriamente
    x_1 = X_1
    sigma_1 = SIGMA_1
    x_2 = X_2
    sigma_2 = SIGMA_2
    p_1 = P_1
    q_1 = Q_1
    p_2 = P_2
    q_2 = Q_2
    # para cada época
    for index in range(quant_epocas):
        # obtendo pontos aleatórios da nossa população
        pontos_aleatorios = random.sample(x, len(x))
        # erro
        e = 0
        # para cada ponto faremos um ajuste
        for ponto in pontos_aleatorios:
            # definição das equações paramétricas
            y_1 = equacao_parametrica(x_1, p_1, q_1)
            y_2 = equacao_parametrica(x_2, p_2, q_2)
            # definição das equações de pertinência
            w_1 = gaussiana(ponto, x_1, sigma_1)
            w_2 = gaussiana(ponto, x_2, sigma_2)
            # estimando o valor
            y = valor_estimado(w_1, w_2, y_1, y_2)
            y_d = round(ponto ** 2, 2)
            # erro
            e += erro(y, y_d)
            # correção dos fatores
            x_1 = ajuste_erro_x_1(x_1, alpha, y, y_d, w_1, w_2, ponto, y_1, y_2, x_1, sigma_1)
            x_2 = ajuste_erro_x_2(x_2, alpha, y, y_d, w_1, w_2, ponto, y_1, y_2, x_2, sigma_2)
            sigma_1 = ajuste_erro_sigma_1(sigma_1, alpha, y, y_d, w_1, w_2, ponto, y_1, y_2, x_1, sigma_1)
            sigma_2 = ajuste_erro_sigma_2(sigma_2, alpha, y, y_d, w_1, w_2, ponto, y_1, y_2, x_2, sigma_2)
            p_1 = ajuste_erro_p_1(p_1, alpha, y, y_d, w_1, w_2, ponto)
            p_2 = ajuste_erro_p_2(p_2, alpha, y, y_d, w_1, w_2, ponto)
            q_1 = ajuste_erro_q_1(q_1, alpha, y, y_d, w_1, w_2)
            q_2 = ajuste_erro_q_2(q_2, alpha, y, y_d, w_1, w_2)
        # ao final faremos o erro médio por década
        erros.append(e / quant_epocas)
    # ao final retornaremos os valores ajustados
    return x_1, sigma_1, x_2, sigma_2, p_1, q_1, p_2, q_2, erros

