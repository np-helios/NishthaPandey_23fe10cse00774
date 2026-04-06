# Basic Text Vectorization Commands

This file explains very basic text vectorization techniques used in NLP for similarity and analysis tasks.

## 1. What Is Text Vectorization?

Text vectorization means converting text into numbers so that a computer can compare and analyze sentences or documents.

Two common techniques are:

- TF-IDF
- Embeddings

## 2. TF-IDF

TF-IDF stands for:

- TF = Term Frequency
- IDF = Inverse Document Frequency

It gives importance to words based on:

- how often they appear in one document
- how rare they are across all documents

### Basic Example

Sentences:

```text
1. I love machine learning
2. I love deep learning
3. Dogs are cute animals
```

### Basic Command

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love machine learning",
    "I love deep learning",
    "Dogs are cute animals"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())
```

### Use

- Find important words in documents
- Compare document similarity
- Use in search and classification tasks

## 3. Cosine Similarity With TF-IDF

Cosine similarity is used to check how similar two text vectors are.

### Basic Command

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(tfidf_matrix)
print(similarity)
```

### Meaning

- Value close to `1` means highly similar
- Value close to `0` means not similar

## 4. Embeddings

Embeddings convert words or sentences into dense numeric vectors.

These vectors capture meaning, so similar words or sentences get similar vectors.

### Example Idea

Words like:

- king
- queen
- man
- woman

can have related vector meanings.

## 5. Basic Embedding Style Command

```python
sentences = [
    "I love machine learning",
    "I enjoy artificial intelligence",
    "The cat is sleeping"
]
```

In GenAI tools, a basic prompt can be:

```text
Convert these sentences into embeddings and compare which two are most similar.
```

### Use

- Semantic similarity
- Recommendation systems
- Clustering similar text
- Searching similar content

## 6. TF-IDF vs Embeddings

TF-IDF:

- Based on word frequency
- Good for simple document comparison
- Easy to understand

Embeddings:

- Based on meaning/context
- Better for semantic similarity
- More powerful in modern GenAI systems

## 7. Similarity Task Example

Texts:

```text
Sentence 1: I like NLP
Sentence 2: I enjoy Natural Language Processing
Sentence 3: The weather is sunny
```

Expected result:

- Sentence 1 and Sentence 2 are more similar
- Sentence 3 is different

## 8. Analysis Tasks Using Vectorization

Text vectorization is used for:

- document similarity
- sentence similarity
- text classification
- clustering
- information retrieval
- semantic search

## 9. Very Simple GenAI Prompt Style

```text
Using TF-IDF, compare these three sentences and tell which two are most similar.
```

```text
Using embeddings, compare these sentences based on meaning.
```

```text
Explain the difference between TF-IDF and embeddings in simple words.
```

## Conclusion

Text vectorization helps convert text into numbers for analysis.

- TF-IDF is simple and frequency-based
- Embeddings are meaning-based and more advanced

Both are useful for similarity and analysis tasks in NLP and GenAI.
