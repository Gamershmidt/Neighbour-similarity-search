# Neighbour-similarity-search
This project aims to create a comparison of neighbors candidates based on their survey answers. 

## Project Structure

The project is structured as follows:

- `search.py`: The main file that connects all components, including the model, metrics (BM25 and FAISS), and dimensionality reduction (PPA-PCA).
- `embeddings.py`: Handles text preprocessing and creation of word-level embeddings using a multilingual model. Initially, five models were compared:
  - `bert-base-multilingual-cased`
  - `distilbert-base-multilingual-cased`
  - `xlm-roberta-base`
  - `xlm-mlm-tlm-xnli15-1024`
  - `camembert-base`
  The best performing model, based on results of comparing matrix of real people's preferences to results on BM25 and FAISS ensemble, was `xlm-roberta-base`.
- `metrics.py`: Implements the BM25 and FAISS ensemble. BM25 focuses on similar words, while FAISS focuses on semantics.
- `ppa-pca.py`: Performs dimensionality reduction, reducing dimensions from 700+ to 50 while capturing 99% of the variance.
