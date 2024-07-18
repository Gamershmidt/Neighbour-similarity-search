import faiss
import numpy as np
from typing import List
from collections import Counter
import math
from .embeddings import tokenize


class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenize(corpus)
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avg_doc_len = 0
        self.doc_freq = {}
        self.term_freq = []
        self.N = len(corpus)

        self._initialize()

    def _initialize(self):
        total_len = 0
        for document in self.corpus:
            terms = document
            self.doc_len.append(len(terms))
            total_len += len(terms)
            term_freq = Counter(terms)
            self.term_freq.append(term_freq)
            for term in term_freq:
                if term not in self.doc_freq:
                    self.doc_freq[term] = 0
                self.doc_freq[term] += 1
        self.avg_doc_len = total_len / self.N

    def idf(self, term: str) -> float:
        return math.log(1 + (self.N - self.doc_freq.get(term, 0) + 0.5) / (self.doc_freq.get(term, 0) + 0.5))

    def score(self, doc_index: int, query: str) -> float:
        terms = query.split()
        score = 0.0
        for term in terms:
            if term not in self.term_freq[doc_index]:
                continue
            tf = self.term_freq[doc_index][term]
            idf = self.idf(term)
            doc_len = self.doc_len[doc_index]
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
        return score


def get_bm25_matrix(corpus: List[str]) -> np.ndarray:
    bm25 = BM25(corpus)
    num_docs = len(corpus)
    bm25_matrix = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            score = bm25.score(i, corpus[j])
            bm25_matrix[i][j] = score
            bm25_matrix[j][i] = score

    return bm25_matrix


def get_faiss_matrix(embeddings) -> np.ndarray:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss_matrix = np.zeros((embeddings.shape[0], embeddings.shape[0]))

    for i in range(embeddings.shape[0]):
        _, indices = index.search(np.array([embeddings[i]]), embeddings.shape[0])
        for j in range(embeddings.shape[0]):
            faiss_matrix[i, indices[0, j]] = 1 / (1 + np.linalg.norm(embeddings[i] - embeddings[indices[0, j]]))

    return faiss_matrix


def ensemble_matrix(queries: List[str], embeddings: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    bm25_matrix = get_bm25_matrix(queries)
    faiss_matrix = get_faiss_matrix(embeddings)

    matrix = alpha * bm25_matrix + (1 - alpha) * faiss_matrix
    np.fill_diagonal(matrix, -1)

    return matrix
