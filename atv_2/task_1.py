import matplotlib.pyplot as plt

from typing import Tuple, List


def load_data() -> Tuple[List[float], List[float]]:
    """Carrega o arquivo data.txt e prepara os dados para serem plotados.

    Returns: Dados definindo a população e o lucro em cada cidade.
    """
    file_path = 'files/data1.txt'
    population: List[float] = []
    income: List[float] = []
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            pop, inc = line.split(',')
            population.append(float(pop))
            income.append(float(inc))

    return population, income


def sub_task_1_2() -> None:
    """Exibe o gráfico dos dados.

    Returns: None.
    """
    population, income = load_data()
    # plotando o gráfico
    plt.plot(population, income, 'ro')
    plt.title('Lucro Obtido dado a população')
    plt.show()


def h_theta(theta_0: float, theta_1: float, x: float) -> float:
    """H de theta dado um valor x irá gerar um valor y estimado. Equação da reta Ax + B.

    Args:
        theta_0: B da equação linear.
        theta_1: A, coeficiente angular.
        x: Valor.

    Returns: Y gerado pela reta.
    """
    return theta_0 + (theta_1 * x)


def cost_function(theta_0: float, theta_1: float, x_sample: List[float], y_sample: List[float],
                  sample_size: int) -> float:
    """Função custo J.

    Args:
        theta_0: Theta 0.
        theta_1: Theta 1.
        x_sample: X da amostra.
        y_sample: Y da amostra.
        sample_size: Tamanho da amostra.

    Returns: Valor da função custo.
    """
    summation: float = 0
    for i in range(sample_size):
        summation += pow(h_theta(theta_0, theta_1, x_sample[i]) - y_sample[i], 2)
    return summation / (2 * sample_size)


def derivate_cost_function_theta_0(theta_0: float, theta_1: float, x_sample: List[float], y_sample: List[float],
                                   sample_size: int) -> float:
    """Derivada da função custo em relação a theta_0.

    Args:
        theta_0: Theta 0.
        theta_1: Theta 1.
        x_sample: X da amostra.
        y_sample: Y da amostra.
        sample_size: Tamanho da amostra.

    Returns: Valor da derivada da função custo em relação a theta 0.
    """
    summation: float = 0
    for i in range(sample_size):
        summation += (h_theta(theta_0, theta_1, x_sample[i]) - y_sample[i])
    return summation / sample_size


def derivate_cost_function_theta_1(theta_0: float, theta_1: float, x_sample: List[float], y_sample: List[float],
                                   sample_size: int) -> float:
    """Derivada da função custo em relação a theta_1.

    Args:
        theta_0: Theta 0.
        theta_1: Theta 1.
        x_sample: X da amostra.
        y_sample: Y da amostra.
        sample_size: Tamanho da amostra.

    Returns: Valor da derivada da função custo em relação a theta 1.
    """
    summation: float = 0
    for i in range(sample_size):
        summation += ((h_theta(theta_0, theta_1, x_sample[i]) - y_sample[i]) * x_sample[i])
    return summation / sample_size


def sub_task_1_3(epoch: int, alpha: float = 0.01) -> Tuple[List[float], float, float]:
    """ Faz a execução da regressão linear utilizando o gradiente descendente.

    Args:
        epoch: Épocas executando o algoritmo.
        alpha: Taxa de aprendizado do algoritmo.

    Returns: None
    """
    # valores iniciais de theta da regressão
    theta_0: float = 1
    theta_1: float = 1
    x_sample, y_sample = load_data()
    sample_size = len(x_sample)
    cost_values: List[float] = []

    for _ in range(epoch):
        temp_0 = theta_0 - (alpha * derivate_cost_function_theta_0(theta_0, theta_1, x_sample, y_sample, sample_size))
        temp_1 = theta_1 - (alpha * derivate_cost_function_theta_1(theta_0, theta_1, x_sample, y_sample, sample_size))
        theta_0 = temp_0
        theta_1 = temp_1
        cost_values.append(cost_function(theta_0, theta_1, x_sample, y_sample, sample_size))
    # pegando o último ajuste

    return cost_values, theta_0, theta_1


# Exibindo o erro por época
cost_values, theta_0, theta_1 = sub_task_1_3(1000)
# epochs = [i for i in range(1, 1000+1)]
# plt.plot(epochs, cost_values)
# plt.title('Função custo em relação a quantidade de épocas')
# plt.show()


# Exibindo o ajuste linear
population, income = load_data()
# plotando o gráfico com os dados obtidos
plt.plot(population, income, 'ro')
linear_adjustment = [h_theta(theta_0, theta_1, x) for x in population]
plt.plot(population, linear_adjustment, 'b')
plt.title('Lucro Obtido dado a população e o ajuste linear')
plt.show()
