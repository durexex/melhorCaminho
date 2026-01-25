# Algoritmo Genetico para TSP

Implementacao de um algoritmo genetico para o problema do caixeiro viajante (TSP) com visualizacao em Pygame e graficos com Matplotlib.

## Requisitos

- Python 3.x

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Como executar

Execucao padrao:

```bash
python tsp.py
```

Executar com limite de geracoes:

```bash
python tsp.py 1000
```

Durante a execucao:
- Feche a janela ou pressione `q` para sair.

## Estrutura do projeto

- `tsp.py`: loop principal, selecao, crossover e mutacao com visualizacao.
- `genetic_algorithm.py`: funcoes do GA (fitness, crossover, mutacao, populacao).
- `draw_functions.py`: funcoes de desenho para Pygame/Matplotlib.
- `benchmark_att48.py`: dados do benchmark att48.
- `demo_crossover.py` e `demo_mutation.py`: demonstracoes isoladas.

## Ajustes rapidos

No arquivo `tsp.py` voce pode ajustar:

- `POPULATION_SIZE`: tamanho da populacao
- `MUTATION_PROBABILITY`: probabilidade de mutacao
- `N_CITIES`: numero de cidades (modo aleatorio)
- `MAX_GENERATION_ALLOWED`: limite de geracoes (via argumento ou padrao)

## Benchmark att48

O bloco do att48 no `tsp.py` esta comentado. Para usar o benchmark:

1) Descomente o bloco "Using att48 benchmark"
2) Ajuste `WIDTH` e `HEIGHT` conforme necessario

