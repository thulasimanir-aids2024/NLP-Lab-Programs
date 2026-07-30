import nltk
from nltk import word_tokenize, pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Get input from user
text = input("Enter a paragraph: ")

# Tokenize the text
tokens = word_tokenize(text)

# Perform POS tagging
tags = pos_tag(tokens)

print("\n========== NAMED ENTITY DETECTION ==========")

count = 0

# Detect proper nouns (NNP) as named entities
for word, tag in tags:
    if tag == "NNP":
        print(word, "-> Named Entity")
        count += 1

# Get actual number of entities
actual = int(input("\nEnter the actual number of named entities: "))

# Calculate accuracy
accuracy = (min(count, actual) / max(count, actual)) * 100

print("\n========== RESULTS ==========")
print("Predicted Named Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")