import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pymorphy3
import torch
from .ppa_pca import PPA_PCA
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
custom_stopwords = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
    'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't",
    'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd",
    "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',
    "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
    'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves',
    'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should',
    "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll",
    "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't",
    'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's",
    'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll",
    "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
}
stopwords_set = set(stopwords.words('english')).union(custom_stopwords).union(stopwords.words('russian'))
stopwords_set.remove("no")
stopwords_set.remove("не")
stopwords_set.remove("нет")


def remove_stopwords(tokens, words):
    tokens_without_stopwords = [token for token in tokens if token.lower() not in stopwords_set]

    return tokens_without_stopwords


class Embeddings:
    def __init__(self, tokens):
        self.docs = tokens

    def get_tokens(self):
        lst = self.docs

        def tokenize(sentence):
            pattern = r"\b\w+(?:'\w+)?\b"
            tokens = re.findall(pattern, sentence)
            return tokens

        tokens = [tokenize(sentence) for sentence in lst]

        filtered_tokens = [remove_stopwords(sentence, stopwords_set) for sentence in tokens]

        lemmatizer = WordNetLemmatizer()

        morph = pymorphy3.MorphAnalyzer()

        def detect_language(text):
            return any(ord(char) > 127 for char in text)

        def multilanguage_lemmatize(text):
            if len(text) == 0:
                return [""]
            if detect_language(text[0]):
                lemmatized_tokens = [morph.parse(token)[0].normal_form for token in text]

            else:
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in text]
            if len(lemmatized_tokens) == 0:
                return [""]
            return lemmatized_tokens

        filtered_tokens = [multilanguage_lemmatize(sentence) for sentence in filtered_tokens]
        for i, hobby in enumerate(filtered_tokens):
            if len(hobby) == 0:
                filtered_tokens[i] = [""]
        return filtered_tokens

    def create_embeddings(self, model, tokenizer):
        tokens = self.get_tokens()
        embeddings_res = []
        for sentence in tokens:
            tokens = tokenizer(sentence, return_tensors='pt', is_split_into_words=True)
            with torch.no_grad():
                outputs = model(**tokens)
                embeddings = outputs.last_hidden_state
                embeddings = embeddings[0, 1, :]
            target_length = 129
            current_length = embeddings.shape[0]
            if current_length < target_length:
                padding_length = target_length - current_length
                zero_padding = torch.zeros((padding_length, embeddings.shape[1]), dtype=torch.float32)
                embeddings_padded = np.array(torch.cat((embeddings, zero_padding), dim=0))
            else:
                embeddings_padded = np.array(embeddings)
            embeddings_res.append(embeddings_padded)
        return PPA_PCA(np.array(embeddings_res), 50)


def tokenize(data):
    return Embeddings(data).get_tokens()
