import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

text = """
The interviewer interviews the president in an interview
"""

# Tokenize
tokens = word_tokenize(text)

# Initialize stemmer
stemmer = PorterStemmer()

# Stem each token
stemmed_tokens = [stemmer.stem(word) for word in tokens]

print(stemmed_tokens)
