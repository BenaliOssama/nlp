import string
import pandas as pd

text = "Remove, this from .? the sentence !!!! !\"#&'()*+,-./:;<=>_"

# Remove all punctuation
punctuation = string.punctuation
cleaned = text.translate(str.maketrans('', '', punctuation))

print(cleaned)
