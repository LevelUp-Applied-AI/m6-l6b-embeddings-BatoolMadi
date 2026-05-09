"""
Module 6 Week B — Stretch: Embedding Space Explorer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel


# ---------------- LOAD GLOVE ----------------

def load_glove(filepath):
    embeddings = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            word = parts[0]
            vector = np.array(parts[1:], dtype=float)

            embeddings[word] = vector

    return embeddings


# ---------------- BERT EMBEDDING ----------------

def extract_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)

    masked = last_hidden * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1)

    embedding = summed / counts

    return embedding.squeeze(0).numpy()


# ---------------- MAIN ----------------

if __name__ == "__main__":

    # Load GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    print(f"Loaded {len(glove)} GloVe vectors")

    # Semantic categories
    categories = {
        "countries": [
            "america", "canada", "china", "india", "france",
            "germany", "italy", "spain", "japan", "brazil"
        ],

        "sports": [
            "football", "soccer", "tennis", "basketball",
            "baseball", "golf", "rugby", "cricket",
            "olympics", "coach"
        ],

        "technology": [
            "computer", "internet", "software", "hardware",
            "database", "network", "server", "python",
            "google", "microsoft"
        ],

        "business": [
            "market", "finance", "economy", "investment",
            "bank", "stock", "trade", "money",
            "profit", "company"
        ],

        "emotions": [
            "happy", "sad", "angry", "fear",
            "joy", "love", "hate", "excited",
            "calm", "anxious"
        ]
    }

    # Expand to 200 words
    all_words = []
    labels = []

    for category, words in categories.items():

        expanded = words * 4  # 10 * 4 = 40 words/category

        for w in expanded:
            if w in glove:
                all_words.append(w)
                labels.append(category)

    # Word vectors
    word_vectors = np.array([glove[w] for w in all_words])

    # t-SNE reduction
    tsne = TSNE(
        n_components=2,
        perplexity=20,
        random_state=42
    )

    reduced_words = tsne.fit_transform(word_vectors)

    # Plot GloVe embeddings
    plt.figure(figsize=(14, 10))

    unique_labels = list(set(labels))

    for label in unique_labels:

        idxs = [i for i, l in enumerate(labels) if l == label]

        plt.scatter(
            reduced_words[idxs, 0],
            reduced_words[idxs, 1],
            label=label
        )

    # Annotate 10 words
    for i in range(min(10, len(all_words))):
        plt.annotate(
            all_words[i],
            (reduced_words[i, 0], reduced_words[i, 1])
        )

    plt.title("t-SNE Visualization of GloVe Word Embeddings")
    plt.legend()
    plt.savefig("plots/glove_tsne.png")
    print("Saved glove_tsne.png")

    # ---------------- BBC NEWS + BERT ----------------

    df = pd.read_csv("data/bbc_news.csv")

    # Pick 20 articles across categories
    selected = []

    for cat in df["category"].unique():
        selected.extend(
            df[df["category"] == cat].head(4).to_dict("records")
        )

    texts = [x["text"] for x in selected]
    text_labels = [x["category"] for x in selected]

    # Load DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )

    model.eval()

    # Document embeddings
    bert_vectors = np.array([
        extract_bert_embedding(t, tokenizer, model)
        for t in texts
    ])

    # PCA reduction
    pca = PCA(n_components=2)

    reduced_docs = pca.fit_transform(bert_vectors)

    # Plot documents
    plt.figure(figsize=(14, 10))

    unique_doc_labels = list(set(text_labels))

    for label in unique_doc_labels:

        idxs = [
            i for i, l in enumerate(text_labels)
            if l == label
        ]

        plt.scatter(
            reduced_docs[idxs, 0],
            reduced_docs[idxs, 1],
            label=label
        )

    # Annotate documents
    for i in range(len(texts)):
        plt.annotate(
            text_labels[i],
            (reduced_docs[i, 0], reduced_docs[i, 1])
        )

    plt.title("PCA Visualization of DistilBERT Document Embeddings")
    plt.legend()
    plt.savefig("plots/bert_pca.png")

    print("Saved bert_pca.png")