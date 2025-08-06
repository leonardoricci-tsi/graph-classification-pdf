# graph-classification-pdf

# 📊 Dataset de Gráficos Científicos

Este diretório contém o conjunto de dados utilizado para o treinamento e validação do modelo de classificação de gráficos científicos.

## 🗂 Estrutura de Diretórios

dataset/
├── bar_chart/
│ ├── bar_chart_001.png
│ ├── bar_chart_002.png
│ └── ...
├── pie_chart/
├── line_graph/
├── scatter_plot/
├── ...


Cada subpasta representa uma classe (categoria) de gráfico, contendo imagens em formato `.png`.

## 📋 Classes

O dataset está organizado nas seguintes categorias:

- `bar_chart` – Gráfico de barras
- `line_graph` – Gráfico de linhas
- `pie_chart` – Gráfico de pizza
- `scatter_plot` – Gráfico de dispersão
- `box_plot` – Boxplot
- `area_chart` – Gráfico de área
- `histogram` – Histograma
- `heatmap` – Mapa de calor
- `other` – Outros tipos (não classificados)

## 🔢 Tamanho do Dataset

- **Total de imagens:** 6.000+
- **Número de classes:** 18
- **Formato das imagens:** `.png`, RGB
- **Resolução padrão:** 224x224 pixels (usado no treinamento com YOLOv8-cls)

## 🧪 Separação

O dataset está separado em:

- `train/` – para treinamento
- `val/` – para validação
- `test/` – para testes manuais e métricas

## 📥 Instruções para uso

1. Copie esta pasta completa para o caminho desejado no seu projeto.
2. Use o caminho `--data graph_dataset_split/graph_dataset_split` ao treinar com YOLOv8.
3. Para adicionar mais gráficos, basta colocar as novas imagens na subpasta da classe correta e reexecutar o script de treino.

## ✍️ Autor

Criado por Leonardo Oliveira – Projeto de Classificação de Gráficos com Visão Computacional e YOLOv8.
Utilização de base de dados CHARTX https://github.com/Alpha-Innovator/ChartVLM
