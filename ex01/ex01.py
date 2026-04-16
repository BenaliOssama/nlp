import pandas as pd

list_ = ["This is my first NLP exercise", "wtf!!!!!"]
series_data = pd.Series(list_, name='text')

# Question 1: Print all texts in lowercase
print(series_data.str.lower())

# Question 2: Print all texts in uppercase
print(series_data.str.upper())
