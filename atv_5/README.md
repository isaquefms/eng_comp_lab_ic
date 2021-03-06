# Atividade 5 de Lab de IC sobre Redes Neurais

## Parte 1
Na parte 1 havia o objetivo de visualizarmos a convergência utilizando o algoritmo `Back Propagation`. Atráves deste algoritmo obtivemos valores consideráveis tanto para a função de perca quanto para a acurácia. Foi possível visualizar que não seriam necessárias mais que 500 épocas para atingir resultados consideráveis. Embora o algoritmo seja simples é possível perceber sua robustez na otimização do problema. Foi possível perceber, também, a capacidade do algoritmo em fazer um ajuste não linear para a obtenção das respostas desejadas.

## Parte 2
Nessa parte do projeto foi possível realizar uma comparação entre os modelos `RandomForest` do `ScikitLearn` e duas redes neurais montadas utilizando o `Keras`. Com a `RandomForest` foi possível obter uma acurácia de `0.771`, resultado relevante porém pouco maior que a média estimada primeiramente. O primeiro modelo de rede neural utlizando o keras com uma camada escodida obteve uma acúracia de `0.766`, valor esse pouco menor que o anterior, exibindo uma certa similaridade entre os modelos. Porém as curvas `roc-auc` mostraram que o `RandomForest` possui um ajuste melhor ao problema utilizando estes parâmetros inicias.

Na segunda parte do projeto houve a proposta da criação de um modelo de 2 camadas escondidas e 6 nós em cada camada, juntamente com uma variáção entre os parâmetros. Algo que causou certa surpresa foi o fato do modelo ter apresentado o melhor ajuste justamente com os primeiros parâmetros descritos na atividade. Demais `solvers` e `learning_rates` não apresentaram resultados satisfatórios estando todos abaixo da primeira configuração.

É importante pontuar que após a geração 400 os resultados não demonstraram ganhos tão relevantes, de modo que poderíamos ter parado o algoritmo na geração 500.
