import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Keywords indicating a medical relationship
keywords = ["treats", "reduces", "controls", "helps"]

# Get input from user
sentence = input("Enter a medical statement: ")
actual = int(input("Enter the actual relation (1 = Present, 0 = Absent): "))

# Tokenize the sentence
tokens = word_tokenize(sentence.lower())

print("\n========== TOKENIZED WORDS ==========")
print(tokens)

# Predict relation based on keywords
predicted = 0

for word in tokens:
    if word in keywords:
        predicted = 1

print("\n========== PREDICTION RESULT ==========")
print("Predicted Relation:", predicted)

# Evaluation
y_true = [actual]
y_pred = [predicted]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n========== EVALUATION METRICS ==========")
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)