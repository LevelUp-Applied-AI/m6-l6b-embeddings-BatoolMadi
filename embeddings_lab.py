"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from transformers import AutoTokenizer, AutoModel


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    return sklearn_cosine(tfidf_matrix)


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}

    with open(filepath, "r", encoding="utf-8") as f:

        for line in f:
            parts = line.strip().split()

            word = parts[0]

            vector = np.array(parts[1:], dtype=float)

            embeddings[word] = vector

    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    words = text.lower().split()

    vectors = [
        embeddings[word]
        for word in words
        if word in embeddings
    ]

    if len(vectors) == 0:
        return np.zeros(50)

    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_hidden = last_hidden * attention_mask

    summed = masked_hidden.sum(dim=1)

    counts = attention_mask.sum(dim=1)

    embedding = summed / counts

    return embedding.squeeze().numpy()


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself.
    """

    results = {}

    # Precompute embeddings
    glove_vectors = np.array([
        text_to_glove(text, glove_embeddings)
        for text in texts
    ])

    bert_vectors = np.array([
        extract_bert_embedding(text, bert_tokenizer, bert_model)
        for text in texts
    ])

    for query in queries:

        query_index = texts.index(query)

        # ---------- TF-IDF ----------
        tfidf_scores = tfidf_sim[query_index]

        tfidf_ranked = sorted(
            [
                (texts[i], tfidf_scores[i])
                for i in range(len(texts))
                if i != query_index
            ],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # ---------- GloVe ----------
        query_glove = glove_vectors[query_index]

        glove_scores = sklearn_cosine(
            [query_glove],
            glove_vectors
        )[0]

        glove_ranked = sorted(
            [
                (texts[i], glove_scores[i])
                for i in range(len(texts))
                if i != query_index
            ],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # ---------- BERT ----------
        query_bert = bert_vectors[query_index]

        bert_scores = sklearn_cosine(
            [query_bert],
            bert_vectors
        )[0]

        bert_ranked = sorted(
            [
                (texts[i], bert_scores[i])
                for i in range(len(texts))
                if i != query_index
            ],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        results[query] = {
            "tfidf": tfidf_ranked,
            "glove": glove_ranked,
            "bert": bert_ranked
        }

    return results


if __name__ == "__main__":

    # Load data
    df = pd.read_csv("data/bbc_news.csv")

    texts = df["text"].tolist()

    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    tfidf_matrix, vectorizer = build_tfidf(texts)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    tfidf_sim = compute_tfidf_similarity(tfidf_matrix)

    print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")

    print(f"Loaded {len(glove)} GloVe vectors")

    sample_emb = text_to_glove(texts[0], glove)

    print(f"Sample GloVe text embedding shape: {sample_emb.shape}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )

    model.eval()

    sample_bert = extract_bert_embedding(
        texts[0],
        tokenizer,
        model
    )

    print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare
    queries = [
        df[df["category"] == cat]["text"].iloc[0]
        for cat in df["category"].unique()
    ]

    comparison = compare_similarities(
        texts,
        queries,
        tfidf_sim,
        glove,
        model,
        tokenizer
    )

    for q in list(comparison.keys())[:2]:

        print(f"\nQuery: {q[:80]}...")

        for method in ["tfidf", "glove", "bert"]:

            top = comparison[q][method]

            print(
                f"  {method}: "
                f"{[t[:40] for t, _ in top]}"
            )