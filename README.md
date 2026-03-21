# Otimizacao de Rotas Medicas - Algoritmo Genetico (TSP/VRP)

Sistema de otimizacao de rotas de entrega de medicamentos e insumos hospitalares utilizando Algoritmos Geneticos. Resolve tanto o problema do Caixeiro Viajante (TSP) quanto o problema de Roteamento de Veiculos com Capacidade (CVRP), com interface interativa em Streamlit.

## Requisitos

- Python 3.10+
- pip

## Configuracao do Ambiente

### 1. Clonar o repositorio

```bash
git clone <url-do-repositorio>
cd melhorCaminho
```

### 2. Criar o ambiente virtual (venv)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Ao ativar o venv, o terminal exibe `(.venv)` no inicio da linha.

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variaveis de ambiente

Copie o arquivo de exemplo e ajuste conforme necessario:

```bash
cp .env.example .env
```

Ou crie manualmente o arquivo `.env` na raiz do projeto. Veja a secao [Parametros de Configuracao](#parametros-de-configuracao) para detalhes.

## Como Executar

### Aplicacao principal (Streamlit)

```bash
o instalastreamlit run tsp.py
```

A interface sera aberta no navegador. Clique em **Play** na sidebar para iniciar o algoritmo genetico.

### Testes automatizados

```bash
python -m pytest tests/ -v
```

### Benchmark VRP

Compara tempo de execucao e qualidade da solucao com 1, 3 e 5 veiculos no dataset ATT-48 (48 cidades):

```bash
python benchmark_vrp.py
```

## Estrutura do Projeto

```
melhorCaminho/
├── .env                    # Variaveis de ambiente (nao versionado)
├── .env.example            # Template das variaveis de ambiente
├── .gitignore              # Arquivos ignorados pelo Git
├── requirements.txt        # Dependencias Python com versoes fixas
├── README.md               # Este arquivo
│
├── tsp.py                  # Aplicacao principal (Streamlit + loop do GA)
├── genetic_algorithm.py    # Core do GA: fitness, crossover, mutacao, split VRP
├── draw_functions.py       # Funcoes de visualizacao (Matplotlib)
├── utils.py                # Relatorio de execucao e dataclasses
├── priority_utils.py       # Utilitarios de prioridade (CSV import/export)
│
├── benchmark_att48.py      # Dados do benchmark ATT-48
├── benchmark_vrp.py        # Benchmark comparativo TSP vs VRP
│
├── demo_crossover.py       # Demonstracao isolada de crossover OX
├── demo_mutation.py        # Demonstracao isolada de mutacao
├── demo_tournament.py      # Demonstracao isolada de selecao por torneio
│
├── tests/
│   ├── test_genetic_algorithm.py   # Testes do GA (fitness, crossover, mutacao)
│   ├── test_priority_utils.py      # Testes de prioridade (CSV parsing)
│   └── test_vrp.py                 # Testes do VRP (split, capacidade, retrocompat.)
│
├── docs/
│   ├── fazer.txt           # Requisitos do projeto
│   └── feito.txt           # Historico de implementacoes
│
├── tasks/                  # Tarefas de desenvolvimento (PRD, Tech Spec)
│   └── prd-capacidade-carga-vrp/
│       ├── prd.md
│       ├── techspec.md
│       ├── tasks.md
│       └── 1_task.md .. 4_task.md
│
└── reports/                # Relatorios gerados automaticamente (nao versionado)
```

## Parametros de Configuracao

Todos os parametros sao configurados via arquivo `.env` na raiz do projeto. Tambem podem ser editados em tempo real na aba **Configuracoes** do Streamlit.

### Gerais

| Parametro | Tipo | Padrao | Descricao |
|---|---|---|---|
| `GERAR_CIDADES` | bool | `False` | `True` gera cidades aleatorias; `False` le de arquivo |
| `NUMBER_OF_CITIES` | int | `20` | Quantidade de cidades (quando `GERAR_CIDADES=True`) |
| `CITIES_LOCATION_FILE` | str | `cities_locations.txt` | Arquivo com coordenadas das cidades |
| `MAX_GENERATION_ALLOWED` | int | `5000` | Total de geracoes do GA |
| `POPULATION_SIZE` | int | `100` | Tamanho da populacao |
| `MUTATION_PROBABILITY` | float | `0.5` | Probabilidade de mutacao |
| `CAR_AUTONOMY` | float | `7000.0` | Autonomia maxima do veiculo (px). `None` desativa |

### ATSP (Custos Assimetricos)

| Parametro | Tipo | Padrao | Descricao |
|---|---|---|---|
| `ATSP_ENABLED` | bool | `True` | Ativa matriz de custos assimetricos |
| `ASYMMETRY_FACTOR` | float | `0.3` | Fator de assimetria (0 = simetrico) |
| `ASYMMETRY_SEED` | int | `42` | Seed para reproducibilidade |

### VRP (Multiplos Veiculos e Capacidade)

| Parametro | Tipo | Padrao | Descricao |
|---|---|---|---|
| `NUM_VEHICLES` | int | `1` | Numero de veiculos (1 = TSP classico) |
| `VEHICLE_CAPACITY_WEIGHT_KG` | float | `500.0` | Capacidade de peso por veiculo (kg). `None` desativa |
| `VEHICLE_CAPACITY_VOLUME_M3` | float | `10.0` | Capacidade de volume por veiculo (m3). `None` desativa |
| `DEMAND_WEIGHT_MIN` | float | `1.0` | Peso minimo de demanda por cidade (kg) |
| `DEMAND_WEIGHT_MAX` | float | `50.0` | Peso maximo de demanda por cidade (kg) |
| `DEMAND_VOLUME_MIN` | float | `0.01` | Volume minimo de demanda por cidade (m3) |
| `DEMAND_VOLUME_MAX` | float | `2.0` | Volume maximo de demanda por cidade (m3) |

## Funcionalidades

### TSP (Caixeiro Viajante)

- Algoritmo Genetico com crossover OX, mutacao swap/inversao e selecao por torneio
- Custos assimetricos (ATSP) com matriz configuravel
- Autonomia limitada do veiculo com retorno ao deposito para reabastecimento
- Prioridades de entrega (medicamentos criticos vs insumos regulares)
- Populacao inicial via heuristica: Nearest Neighbours, Greedy Multi-Fragment ou Convex Hull
- Mutacao adaptativa com deteccao de estagnacao

### VRP (Roteamento de Veiculos)

- Multiplos veiculos com rotas independentes
- Restricao de capacidade por peso (kg) e volume (m3)
- Greedy Split para divisao otima de rotas
- Retrocompativel: `NUM_VEHICLES=1` sem capacidade = TSP puro
- Demanda por cidade gerada automaticamente (modulada por prioridade)
- Visualizacao com cores distintas por veiculo

### Interface Streamlit

- **Aba Mapa**: visualizacao em tempo real da evolucao do GA e rotas no mapa
- **Aba Mapa de Calor**: heatmap da matriz de custos assimetricos
- **Aba Configuracoes**: edicao de todos os parametros do `.env`
- **Aba Prioridades**: gestao de prioridades e demandas por cidade
- **Sidebar**: parametros VRP, controle Play, e resultado detalhado por veiculo

## Desativando o venv

Quando terminar de trabalhar no projeto:

```bash
deactivate
```
