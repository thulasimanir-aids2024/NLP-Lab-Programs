import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Download required datasets
nltk.download('wordnet')
nltk.download('omw-1.4')

# Store text documents
documents = []

# Get input from user
n = int(input("Enter the number of text documents: "))

for i in range(n):
    text = input(f"Enter document {i+1}: ")
    documents.append(text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Cosine Similarity
print("\n========== COSINE SIMILARITY MATRIX ==========")
similarity_matrix = cosine_similarity(X)
print(similarity_matrix)

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X)

print("\n========== DOCUMENT CLUSTERS ==========")
for i in range(len(documents)):
    print(f"Document {i+1}: {documents[i]}")
    print("Cluster:", kmeans.labels_[i])
    print()

# WordNet Similarity
word1 = input("Enter the first word: ")
word2 = input("Enter the second word: ")

synsets1 = wn.synsets(word1)
synsets2 = wn.synsets(word2)

print("\n========== WORDNET SIMILARITY ==========")

if synsets1 and synsets2:
    similarity = synsets1[0].path_similarity(synsets2[0])
    print(f"Similarity between '{word1}' and '{word2}':", similarity)
else:
    print("Unable to calculate similarity because one or both words were not found in WordNet.")