# Algoritmo Genetico para TSP

Implementacao de um algoritmo genetico para o problema do caixeiro viajante (TSP) com interface em Streamlit e graficos em Matplotlib.

## Requisitos

- Python 3.10+
- pip

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Como executar

Execucao padrao:

```bash
streamlit run tsp.py
```

Durante a execucao:
- A pagina e atualizada conforme as geracoes.
- Um relatorio e gerado em `reports\YYYYMMDDHHMMSS.txt`.

## Arquivos de dados

Quando `GERAR_CIDADES = False`, o script le:
- `cities_locations.txt`: uma cidade por linha no formato `x,y`.
- `cities_asynmetric_cost_matrix.txt`: matriz de custos assimetricos (opcional).

Se `ATSP_ENABLED = True` e a matriz nao existir, ela sera criada automaticamente.

## Ajustes rapidos (tsp.py)

- `GERAR_CIDADES` e `NUMBER_OF_CITIES`: gera cidades aleatorias.
- `ATSP_ENABLED` e `ASYMMETRY_FACTOR`: ativa custos assimetricos.
- `POPULATION_SIZE`, `MUTATION_PROBABILITY`, `JUST_SWAP`: parametros do GA.
- `MAX_GENERATION_ALLOWED` e `RENDER_EVERY`: controle de execucao.
- `ONLY_RANDOM_POPULATION`, `RANDOM_POPULATION_PERCENT`, `GENERATE_POLPULATION_*`: geracao da populacao inicial.

## Estrutura do projeto

- `tsp.py`: loop principal e UI em Streamlit.
- `genetic_algorithm.py`: funcoes do GA (fitness, crossover, mutacao, populacao).
- `draw_functions.py`: funcoes de desenho para Matplotlib/Streamlit.
- `utils.py`: relatorio de execucao.
- `benchmark_att48.py`: dados do benchmark att48.
- `demo_crossover.py`, `demo_mutation.py`, `demo_tournament.py`: demonstracoes isoladas.

## Benchmark att48

O bloco do att48 no `tsp.py` esta comentado. Para usar:

1. Descomente o bloco "Using att48 benchmark".
2. Ajuste `WIDTH` e `HEIGHT` conforme necessario.
