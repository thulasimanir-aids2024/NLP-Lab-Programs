import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Get input from user
text = input("Enter a paragraph: ")

# Tokenize the input
tokens = word_tokenize(text)

# Perform POS tagging
tagged_words = pos_tag(tokens)

# Display original text
print("\nOriginal Input:")
print(text)

# Display tokenized words
print("\nTokenized Words:")
print(tokens)

# Display POS tagging results
print("\nPart-of-Speech (POS) Tags:")
for word, tag in tagged_words:
    print(word, "->", tag)

# Display common POS tag meanings
print("\nCommon POS Tag Meanings:")
print("NN  -> Noun")
print("NNS -> Plural Noun")
print("NNP -> Proper Noun")
print("VB  -> Verb")
print("VBD -> Verb (Past Tense)")
print("VBG -> Verb (Present Participle)")
print("JJ  -> Adjective")
print("RB  -> Adverb")
print("PRP -> Pronoun")
print("DT  -> Determiner")

# Display total number of tokens
print("\nTotal Number of Tokens:", len(tokens))