import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

text = """01 Edu System presents an innovative curriculum in software engineering and programming. With a renowned industry-leading reputation, the curriculum has been rigorously designed for learning skills of the digital world and technology industry. Taking a different approach than the classic teaching methods today, learning is facilitated through a collective and co-creative process in a professional environment."""

result = preprocess_text(text)
print(result)
