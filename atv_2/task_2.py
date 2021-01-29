import matplotlib.pyplot as plt

from math import sqrt
from typing import Tuple, List

import numpy


def load_data() -> Tuple[List[int], List[int], List[float]]:
    """Carrega o arquivo data2.txt e prepara os dados para serem utilizados.

    Returns: Dados definindo o tamanho da casa o número de quartos e o preço da casa.
    """
    file_path = 'files/data2.txt'
    house_length: List[int] = []
    rooms_number: List[int] = []
    houses_price: List[float] = []
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            h_l, r_n, h_p = line.split(',')
            house_length.append(int(h_l))
            rooms_number.append(int(r_n))
            houses_price.append(float(h_p))

    return house_length, rooms_number, houses_price


def standard_deviation(samples: List[int], mean: float) -> float:
    """Calcula o desvio padrão de uma amostra.

    Args:
        samples: Amostra.
        mean: Média.

    Returns: Desvio padrão.
    """
    sample_length: float = len(samples)
    summation: float = 0
    for sample in samples:
        summation += pow(sample - mean, 2)
    return sqrt(summation/sample_length)


def sub_task_2_1(house_length: List[int], rooms_number: List[int]) -> Tuple[List[float], List[float]]:
    """Realiza a normalização dos valores da feature.

    Args:
        house_length: Tamanho da casa.
        rooms_number: Número de quartos.

    Returns: Features normalizadas.
    """
    h_l_mean: float = 0
    r_n_mean: float = 0
    features_length = len(house_length)
    # iterando para descobrimos os valores médios
    for i in range(features_length):
        h_l_mean += house_length[i]
        r_n_mean += rooms_number[i]
    # calculando as médias
    h_l_mean = h_l_mean / features_length
    r_n_mean = r_n_mean / features_length
    # cálculo do desvio padrão
    h_l_standard_d = standard_deviation(house_length, h_l_mean)
    r_n_standard_d = standard_deviation(rooms_number, r_n_mean)
    # ajuste das amostras
    house_length = [round((element - h_l_mean)/h_l_standard_d, 2) for element in house_length]
    rooms_number = [round((element - r_n_mean)/r_n_standard_d, 2) for element in rooms_number]
    return house_length, rooms_number


def h_theta(theta_0: float, theta_1: float, theta_2: float, feat_x_0: float, feat_x_1: float) -> float:
    """H de theta dado um valor x irá gerar um valor y estimado. Equação do plano C + Bx_1 + Cx_2.

    Args:
        theta_0: C da equação.
        theta_1: B da equação.
        theta_2: A da equação.
        feat_x_0: Valor da primeira feat.
        feat_x_1: Valor da segunda feat.

    Returns: Y gerado pelo plano.
    """
    return theta_0 + (theta_1 * feat_x_0) + (theta_2 * feat_x_1)


def cost_function(theta_0: float, theta_1: float, theta_2: float, x_0_sample: List[float], x_1_sample: List[float],
                  y_sample: List[int], sample_size: int) -> float:
    """Função custo J.

    Args:
        theta_0: Theta 0.
        theta_1: Theta 1.
        theta_2: Theta 2.
        x_0_sample: X_0 da amostra.
        x_1_sample: X_1 da amostra.
        y_sample: Y da amostra.
        sample_size: Tamanho da amostra.

    Returns: Valor da função custo.
    """
    summation: float = 0
    for i in range(sample_size):
        summation += pow(h_theta(theta_0, theta_1, theta_2, x_0_sample[i], x_1_sample[i]) - y_sample[i], 2)
    return summation / (2 * sample_size)


def derivate_cost_function_theta_0(theta_0: float, theta_1: float, theta_2: float, x_0_sample: List[float],
                                   x_1_sample: List[float], y_sample: List[float], sample_size: int) -> float:
    """Derivada da função custo em relação a theta_0.

    Args:
        theta_0: Theta 0.
        theta_1: Theta 1.
        theta_2: Theta 2.
        x_0_sample: X_0 da amostra.
        x_1_sample: X_1 da amostra.
        y_sample: Y da amostra.
        sample_size: Tamanho da amostra.

    Returns: Valor da derivada da função custo em relação a theta 0.
    """
    summation: float = 0
    for i in range(sample_size):
        summation += (h_theta(theta_0, theta_1, theta_2, x_0_sample[i], x_1_sample[i]) - y_sample[i])
    return summation / sample_size


def derivate_cost_function_theta_1(theta_0: float, theta_1: float, theta_2: float, x_0_sample: List[float],
                                     x_1_sample: List[float], y_sample: List[float], sample_size: int) -> float:
    """Derivada da função custo em relação a theta_1.

    Args:
        theta_0: Theta 0.
        theta_1: Theta 1.
        theta_2: Theta 2.
        x_0_sample: X da amostra.
        x_1_sample: X da amostra.
        y_sample: Y da amostra.
        sample_size: Tamanho da amostra.

    Returns: Valor da derivada da função custo em relação a theta 1.
    """
    summation: float = 0
    for i in range(sample_size):
        summation += ((h_theta(theta_0, theta_1, theta_2, x_0_sample[i], x_1_sample[i]) - y_sample[i]) * x_0_sample[i])
    return summation / sample_size


def derivate_cost_function_theta_2(theta_0: float, theta_1: float, theta_2: float, x_0_sample: List[float],
                                   x_1_sample: List[float], y_sample: List[float], sample_size: int) -> float:
    """Derivada da função custo em relação a theta_1.

    Args:
        theta_0: Theta 0.
        theta_1: Theta 1.
        theta_2: Theta 2.
        x_0_sample: X da amostra.
        x_1_sample: X da amostra.
        y_sample: Y da amostra.
        sample_size: Tamanho da amostra.

    Returns: Valor da derivada da função custo em relação a theta 1.
    """
    summation: float = 0
    for i in range(sample_size):
        summation += ((h_theta(theta_0, theta_1, theta_2, x_0_sample[i], x_1_sample[i]) - y_sample[i]) * x_1_sample[i])
    return summation / sample_size


# gradiente descendente
def sub_task_2_2(epoch: int, alpha: float = 0.01) -> Tuple[List[float], float, float, float]:
    """ Faz a execução da regressão linear utilizando o gradiente descendente.

    Args:
        epoch: Épocas executando o algoritmo.
        alpha: Taxa de aprendizado do algoritmo.

    Returns: None
    """
    # valores iniciais de theta da regressão
    theta_0: float = 1
    theta_1: float = 1
    theta_2: float = 1
    x_0_sample, x_1_sample, y_sample = load_data()
    x_0_sample, x_1_sample = sub_task_2_1(x_0_sample, x_1_sample)
    sample_size = len(x_0_sample)
    cost_values: List[float] = []

    for _ in range(epoch):
        temp_0 = theta_0 - (alpha * derivate_cost_function_theta_0(theta_0, theta_1, theta_2, x_0_sample, x_1_sample,
                                                                   y_sample, sample_size))
        temp_1 = theta_1 - (alpha * derivate_cost_function_theta_1(theta_0, theta_1, theta_2, x_0_sample, x_1_sample,
                                                                   y_sample, sample_size))
        temp_2 = theta_2 - (alpha * derivate_cost_function_theta_2(theta_0, theta_1, theta_2, x_0_sample, x_1_sample,
                                                                   y_sample, sample_size))
        theta_0 = temp_0
        theta_1 = temp_1
        theta_2 = temp_2
        cost_values.append(cost_function(theta_0, theta_1, theta_2, x_0_sample, x_1_sample, y_sample, sample_size))
    # pegando o último ajuste

    return cost_values, theta_0, theta_1, theta_2


def task_3() -> None:
    """Calcula os valores usando a equação normal.

    Returns: None.
    """
    # inicializando para montarmos as matrizes
    x_1_sample, x_2_sample, y_sample = load_data()
    # transformando as listas em matrizes
    X = []
    for index, _ in enumerate(x_1_sample):
        X.append([1, x_1_sample[index], x_2_sample[index]])

    # montando a matriz X
    X = numpy.asmatrix(X)
    y_sample = numpy.array(y_sample).reshape(len(y_sample), 1)
    # operando
    first_factor = numpy.linalg.inv(numpy.matmul(X.T, X))
    second_factor = numpy.matmul(first_factor, X.T)
    theta = numpy.matmul(second_factor, y_sample)
    return theta


thetas = task_3()
# Exibindo o erro por época
cost_values, theta_0, theta_1, theta_2 = sub_task_2_2(1000)
print(theta_0, theta_1, theta_2)
print(thetas[0][0], thetas[1][0], thetas[2][0])
# epochs = [i for i in range(1, 1000+1)]
# plt.plot(epochs, cost_values)
# plt.title('Função custo em relação a quantidade de épocas. Alpha = 0.1')
# plt.show()
