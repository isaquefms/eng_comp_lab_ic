# Tarefa Regressão Logística - Parte 2
## Daniel Santana e Isaque Fernando

### Questão 4
A questão 4 tem por objetivo simplesmente treinar modelos diferentes de regressão logística afim de comparar os diferentes algoritmos de penalidade e os diferentes solvers. Devido a demora no processamento dos modelos optamos em deixar o `l2` apenas com o solver padrão.

### Questão 5
A questão 5 faz uma comparação entre as magnitude dos coeficientes das regressões lineares feitas. Visualizando vemos que eles se encontram no range de [-10, 10]. Pelos gráficos percebemos que a `l2` possui um desvio padrão menor que os demais métodos de penalização. Isso se deve a forma como a correção é feita que leva em consideração o quadrado de `theta`.

### Questão 6
Nessa questão temos a predição de cada classe e a probabilidade para cada classe. Vemos que para todos os modelos e classes temos uma probabilidade acima dos 99%. Ao analisar essa métrica temos a impressão de que está sendo avaliada apenas uma classe por modelo.

### Questão 7
Na questão 7 há o cálculo das principais métricas de avaliação do modelo. Os resultados mostram que os modelos são robustos e possuem um bom aprendizado, sendo que do `lr` para o `l2` temos um acréscimo pequeno de qualidade, na ordem da terceira casa decimal, para todas as métricas analisadas.

### Questão 8
A matriz de confusão para os modelos exibiu uma boa concentração na diagonal, refletindo as métricas da questão anterior. Algo interessante de se notar é o fato de até mesmos as previsões incorretas não estarem tão distantes das corretas. Isso é visível ao notar que os valores incorretos também estão próximos da diagonal.

### Questão 9
Apresenta erro no código.
