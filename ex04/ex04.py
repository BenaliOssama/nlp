import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = """
The goal of this exercise is to learn to remove stop words with NLTK.  Stop words usually refers to the most common words in a language.
"""

# Tokenize
tokens = word_tokenize(text)

# Get English stop words
stop_words = set(stopwords.words('english'))

# Remove stop words
filtered_tokens = [word for word in tokens if word not in stop_words]

print(filtered_tokens)
