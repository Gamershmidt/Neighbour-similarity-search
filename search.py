from modules.metrics import ensemble_matrix
from modules.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import pymongo
import numpy as np
import os


def mongo_db():
    client = pymongo.MongoClient("YOUR MONGODB PASSWORD")

    db = client.get_database("dls")
    collection = db.get_collection("text-questions")
    documents = collection.find()
    dataset = []
    for document in documents:
        dataset.append([])
        for i in range(len(document['answers'])):
            dataset[-1].append(document['answers'][i]["value"])
    return dataset


def get_most_similar_answers(_id, dataset, matrix):
    user_vector = matrix[_id]
    similarities = []
    for i in range(user_vector.shape[0]):
        if i != _id:
            similarities.append((i, user_vector[i]))
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_answers = []

    for i in similarities:
        similar_answers.append((dataset[i[0]], i[0]))

    return similar_answers[:5]


def print_most_similar(_id, dataset, matrix):
    similar_answers = get_most_similar_answers(_id, dataset, matrix)
    print(f"User {_id} answer: {dataset[_id]}")
    print("-" * 50)
    print("Most similar answers:")
    for i in similar_answers:
        print(f"User {i[1]}: {i[0]}")
    print("-" * 50)


if __name__ == '__main__':
    dataset = mongo_db()
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")
    for i in range(2):
        filename = os.path.join("../results", f"result_{i}.txt")
        embeddings = Embeddings(dataset[i]).create_embeddings(model, tokenizer)
        matrix = ensemble_matrix(dataset[i], embeddings)
        np.savetxt(filename, matrix, fmt='%.7f')

        print_most_similar(0, dataset[i], matrix)
