# Stretch Analysis — Embedding Space Explorer

## GloVe Word Embedding Visualization

The t-SNE visualization shows that semantically related words tend to cluster together in the embedding space. Technology terms such as “computer”, “software”, and “internet” appear close to each other, while sports-related words form a separate cluster. Emotion words are more isolated because they are used in different contexts than topical news vocabulary. Some overlap appears between business and country-related terms due to shared geopolitical and economic context.

## DistilBERT Document Embedding Visualization

The PCA projection of BBC News article embeddings shows that DistilBERT captures topic-level relationships between documents. Politics and business articles partially overlap because many news stories discuss economics and government decisions together. Sports and entertainment articles form more separated groups. A few outlier documents appear between clusters, likely because they contain mixed-topic content or shared named entities.

## Overall Observation

The visualizations demonstrate the difference between static and contextual embeddings. GloVe captures fixed semantic relationships between words, while DistilBERT captures contextual meaning at the document level. Dimensionality reduction makes these relationships visible and helps explain how embedding models organize semantic information in high-dimensional space.