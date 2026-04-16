import nltk
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize, word_tokenize

text = """Bitcoin is a cryptocurrency invented in 2008 by an unknown person or group of people using the name Satoshi Nakamoto. The currency began use in 2009 when its implementation was released as open-source software."""

# Question 1: Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:")
print(sentences)

# Question 2: Word tokenization
words = word_tokenize(text)
print("\nWords:")
print(words)
