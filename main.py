from pdf_loader import load_documents
from embedding import embedding
from preprocessing import preprocess
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def rank_documents(input_doc, documents):
    documents = np.insert(documents, 0, input_doc)
    preprocessed_documents = preprocess(documents)
    print("Encoding with BERT...")
    documents_vectors = embedding(preprocessed_documents)
    print("Encoding finished")
    print(documents_vectors.shape)

    pairwise = cosine_similarity(documents_vectors)

    print('Resume ranking:')

    sorted_idx = np.argsort(pairwise[0])[::-1]

    for idx in sorted_idx[:10]:
        if idx == 0:
            continue
        print(f'Resume of candidite {idx}')
        print(f'Cosine Similarity: {pairwise[0][idx]}\n')


if __name__ == '__main__':
    rank_documents('I want a data scientist',
                   load_documents('documents'))
