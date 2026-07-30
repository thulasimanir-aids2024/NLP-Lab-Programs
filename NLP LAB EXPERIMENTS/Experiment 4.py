import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Store text passages
documents = []

# Get number of documents
n = int(input("Enter the number of text passages: "))

# Read documents
for i in range(n):
    text = input(f"Enter text passage {i+1}: ")
    documents.append(text)

# Search query
query = input("\nEnter the search keyword or sentence: ")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Convert query into TF-IDF vector
query_vector = vectorizer.transform([query])

# Cosine Similarity using TF-IDF
tfidf_scores = cosine_similarity(query_vector, X)

print("\n========== TF-IDF SIMILARITY SCORES ==========")
for i, score in enumerate(tfidf_scores[0]):
    print(f"Text Passage {i+1}: {round(score, 3)}")

# Apply LSA using SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vector)

# Cosine Similarity in LSA space
lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\n========== LSA SIMILARITY SCORES ==========")
for i, score in enumerate(lsa_scores[0]):
    print(f"Text Passage {i+1}: {round(score, 3)}")

# Find the most relevant document
best_match = np.argmax(lsa_scores)

print("\n========== MOST RELEVANT TEXT ==========")
print(documents[best_match])