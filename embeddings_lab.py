"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


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
            vec = np.array(parts[1:], dtype=float)
            embeddings[word] = vec
    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]

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
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state  # (1, seq_len, 768)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)

    masked = last_hidden * attention_mask
    summed = masked.sum(dim=1)
    counts = attention_mask.sum(dim=1)

    embedding = summed / counts
    return embedding.squeeze(0).numpy()


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself. Return:

        {query_text: {"tfidf": [(text, score), ...],
                      "glove": [(text, score), ...],
                      "bert":  [(text, score), ...]}}
    """
    results = {}

    # Precompute embeddings
    glove_vecs = [text_to_glove(t, glove_embeddings) for t in texts]

    bert_vecs = [
        extract_bert_embedding(t, bert_tokenizer, bert_model)
        for t in texts
    ]

    glove_sim = sklearn_cosine(glove_vecs)
    bert_sim = sklearn_cosine(bert_vecs)

    for q in queries:
        if q not in texts:
            continue

        q_idx = texts.index(q)

        def top3(sim_matrix):
            scores = list(enumerate(sim_matrix[q_idx]))
            scores = [
                (texts[i], s)
                for i, s in scores
                if i != q_idx
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:3]

        results[q] = {
            "tfidf": top3(tfidf_sim),
            "glove": top3(glove_sim),
            "bert": top3(bert_sim),
        }

    return results

# Tier 1
def l2_normalize(vectors):
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


def euclidean_distance_matrix(vectors):
    n = len(vectors)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(vectors[i] - vectors[j])

    return dist

# Tier 2
def analyze_tokenization(texts, tokenizer):
    word_counts = []
    subword_splits = {}

    for text in texts:
        words = text.split()
        bert_tokens = tokenizer.tokenize(text)

        word_counts.append(len(bert_tokens) / max(len(words), 1))

        for w in words:
            tokens = tokenizer.tokenize(w)
            if len(tokens) > 1:
                subword_splits[w] = subword_splits.get(w, 0) + 1

    avg_subwords = np.mean(word_counts)

    top_subwords = sorted(subword_splits.items(), key=lambda x: x[1], reverse=True)[:10]

    return avg_subwords, top_subwords

# Tier 3
def precision_at_k(ranked_list, relevant_set, k):
    top_k = ranked_list[:k]
    return sum(1 for x in top_k if x in relevant_set) / k


def reciprocal_rank(ranked_list, relevant_set):
    for i, item in enumerate(ranked_list):
        if item in relevant_set:
            return 1 / (i + 1)
    return 0


def evaluate_system(rankings, relevance_dict):
    results = {
        "tfidf": {"mrr": 0, "p3": 0, "p5": 0},
        "glove": {"mrr": 0, "p3": 0, "p5": 0},
        "bert": {"mrr": 0, "p3": 0, "p5": 0},
    }

    n = len(rankings)

    for query, methods in rankings.items():
        relevant = set(relevance_dict[query])

        for method in ["tfidf", "glove", "bert"]:
            ranked = [x[0] for x in methods[method]]

            results[method]["mrr"] += reciprocal_rank(ranked, relevant)
            results[method]["p3"] += precision_at_k(ranked, relevant, 3)
            results[method]["p5"] += precision_at_k(ranked, relevant, 5)

    for m in results:
        for k in results[m]:
            results[m][k] /= n

    return results

if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModel

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
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare
    queries = [
        df[df["category"] == cat]["text"].iloc[0]
        for cat in df["category"].unique()
    ]

    comparison = compare_similarities(
        texts, queries, tfidf_sim, glove, model, tokenizer
    )

    for q in list(comparison.keys())[:2]:
        print(f"\nQuery: {q[:80]}...")
        for method in ["tfidf", "glove", "bert"]:
            top = comparison[q][method]
            print(f"  {method}: {[t[:40] for t, _ in top]}")