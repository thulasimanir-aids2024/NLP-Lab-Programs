import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.tag import hmm
from nltk.corpus import treebank

# Download required datasets
nltk.download('punkt')
nltk.download('treebank')

# -----------------------------
# Input Text
# -----------------------------
text = input("Enter a text passage: ")

# Tokenization
tokens = nltk.word_tokenize(text.lower())

print("\n========== TOKENIZED WORDS ==========")
print(tokens)

# -----------------------------
# N-Gram Language Model
# -----------------------------
print("\n========== N-GRAM LANGUAGE MODEL ==========")

# Unigrams
unigrams = list(ngrams(tokens, 1))
print("\nUnigrams:")
print(unigrams)

# Bigrams
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:")
print(bigrams)

# Trigrams
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:")
print(trigrams)

# Word Frequency
fd = FreqDist(tokens)

print("\nWord Frequency Distribution:")
for word, freq in fd.items():
    print(word, ":", freq)

# -----------------------------
# Hidden Markov Model (HMM)
# -----------------------------
print("\n========== HIDDEN MARKOV MODEL (HMM) ==========")

# Train HMM using the Treebank corpus
train_data = treebank.tagged_sents()[:3000]

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Predict POS tags
tagged_sentence = hmm_tagger.tag(tokens)

print("\nPredicted Part-of-Speech Tags:")
for word, tag in tagged_sentence:
    print(word, "->", tag)

# -----------------------------
# Comparison
# -----------------------------
print("\n========== COMPARISON ==========")

print("\nN-Gram Language Model")
print("- Identifies patterns of consecutive words.")
print("- Predicts the next word using previous words.")
print("- Commonly used in text prediction and language modeling.")

print("\nHidden Markov Model (HMM)")
print("- Predicts the Part-of-Speech (POS) tag for each word.")
print("- Uses transition and emission probabilities.")
print("- Commonly used in POS tagging and sequence labeling.")

print("\nSummary")
print("N-Gram analyzes word sequences, whereas HMM predicts hidden states such as POS tags.")