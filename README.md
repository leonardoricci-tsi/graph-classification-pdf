# 📊 graph-classification-pdf + Agro Insights

Este repositório contém dois blocos principais:

1. **Dataset e modelo** para classificação de gráficos científicos (baseado em YOLOv8-cls).
2. **Pipeline Agro** para aplicar o modelo em relatórios agrícolas (PDFs da Embrapa, Conab, USDA etc.) e gerar insights automáticos.

---

## 🗂 Estrutura de Diretórios

graph-classification-pdf/
├── dataset/ # Dataset de gráficos científicos
├── weights/ # Pesos treinados (ex.: best.pt)
├── agro_pipeline/ # Scripts de extração de insights agro
│ ├── run_agro_insights.py
│ └── requirements_agro.txt
├── agro_outputs/ # Saídas (insights.csv, relatorio.md, recortes .png)
└── pdfs/ # PDFs de teste (não versionados no Git)


---

## 📋 Classes do Dataset

O dataset está organizado nas seguintes categorias principais:

- `bar_chart` – Gráfico de barras  
- `line_graph` / `line_chart` – Gráfico de linhas  
- `pie_chart` – Gráfico de pizza  
- `scatter_plot` – Gráfico de dispersão  
- `box_plot` – Boxplot  
- `area_chart` – Gráfico de área  
- `histogram` – Histograma  
- `heatmap` – Mapa de calor  
- `map` – Mapas (NDVI, pluviometria, etc.)  
- `candlestick` – Gráfico de velas (financeiro)  
- `other` – Outros tipos

> ⚠️ As classes podem variar conforme o dataset usado no treino.  
> O modelo atual (`best.pt`) foi treinado em 18 classes do **CHARTX**: https://github.com/Alpha-Innovator/ChartVLM

---

## 🔢 Tamanho do Dataset

- **Total de imagens:** 6.000+  
- **Número de classes:** 18  
- **Formato:** `.png`, RGB  
- **Resolução:** 224x224 px (treino YOLOv8-cls)  

Separação:
- `train/` – treino  
- `val/` – validação  
- `test/` – testes manuais  

---

## 🌱 Pipeline Agro (Insights Automáticos)

Com o script `agro_pipeline/run_agro_insights.py`, você pode rodar PDFs técnicos e obter:

- **Classificação de gráficos** página a página.  
- **Agrupamento por tópicos** relevantes ao agro:
  - NDVI / Índices de vegetação  
  - Clima / Chuvas / Temperatura  
  - Mercado / Preços / Commodities  
  - Produção / Rendimento  
  - Sanidade / Doenças / Pragas  
- **Relatório automático** em Markdown (`relatorio.md`).  
- **Tabela CSV** (`insights.csv`) com classe, confiança, tópico e resumo OCR.  
- **Recortes `.png`** dos gráficos detectados.

### ▶️ Como rodar

1. Ative o ambiente virtual:
   ```bash
   source .venv311/bin/activate
pip install -r agro_pipeline/requirements_agro.txt
brew install tesseract

2. Pipeline:
python agro_pipeline/run_agro_insights.py \
  --pdfs "pdfs/AGRO 34.pdf" \
  --weights weights/best.pt \
  --conf 0.35

