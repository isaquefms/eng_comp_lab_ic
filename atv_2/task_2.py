from math import sqrt
from typing import Tuple, List


def load_data() -> Tuple[List[int], List[int], List[int]]:
    """Carrega o arquivo data2.txt e prepara os dados para serem utilizados.

    Returns: Dados definindo o tamanho da casa o número de quartos e o preço da casa.
    """
    file_path = 'files/data2.txt'
    house_length: List[int] = []
    rooms_number: List[int] = []
    houses_price: List[int] = []
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            h_l, r_n, h_p = line.split(',')
            house_length.append(int(h_l))
            rooms_number.append(int(r_n))
            houses_price.append(int(h_p))

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


def sub_task_2_1(house_length: List[int], rooms_number: List[int],
                 houses_prices: List[int]) -> Tuple[List[float], List[float], List[float]]:
    """Realiza a normalização dos valores da feature.

    Args:
        house_length: Tamanho da casa.
        rooms_number: Número de quartos.
        houses_prices: Valor das casas.

    Returns: Features normalizadas.
    """
    h_l_mean: float = 0
    r_n_mean: float = 0
    h_p_mean: float = 0
    features_length = len(house_length)
    # iterando para descobrimos os valores médios
    for i in range(features_length):
        h_l_mean += house_length[i]
        r_n_mean += rooms_number[i]
        h_p_mean += houses_prices[i]
    # calculando as médias
    h_l_mean = h_p_mean / features_length
    r_n_mean = r_n_mean / features_length
    h_p_mean = h_p_mean / features_length
    # cálculo do desvio padrão
    h_l_standard_d = standard_deviation(house_length, h_l_mean)
    r_n_standard_d = standard_deviation(rooms_number, r_n_mean)
    h_p_standard_d = standard_deviation(houses_prices, h_p_mean)
    # ajuste das amostras
    house_length = [(element - h_l_mean)/h_l_standard_d for element in house_length]
    rooms_number = [(element - r_n_mean)/r_n_standard_d for element in rooms_number]
    houses_prices = [(element - h_p_mean)/h_p_standard_d for element in houses_prices]
    return house_length, rooms_number, houses_prices


a, b, c = load_data()
a, b, c = sub_task_2_1(a, b, c)
print(a)
print(b)
print(c)
