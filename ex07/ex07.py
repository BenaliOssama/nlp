import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Load tweets data
with open('resources/tweets_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Separate tweets and labels
tweets = []
labels = []
for line in lines:
    parts = line.strip().split(', ', 1)  # Split on first comma+space only
    if len(parts) == 2:
        label_str, tweet = parts
        # Map label: positive=1, negative=-1, neutral=0
        if label_str.lower() == 'positive':
            labels.append(1)
        elif label_str.lower() == 'negative':
            labels.append(-1)
        else:
            labels.append(0)
        tweets.append(tweet)

print(f"Loaded {len(tweets)} tweets with {len(labels)} labels")
print(f"Label distribution: {pd.Series(labels).value_counts()}")

# Question 1: Create CountVectorizer
vectorizer = CountVectorizer(max_features=500, lowercase=True, stop_words='english')
count_matrix = vectorizer.fit_transform(tweets)

print("\nQuestion 1 - Shape of count matrix:")
print(count_matrix)

# Question 2: Create DataFrame from sparse matrix
count_vectorized_df = pd.DataFrame.sparse.from_spmatrix(count_matrix)
count_vectorized_df.columns = vectorizer.get_feature_names_out()

print("\nQuestion 2 - First 3 rows, columns 400-402:")
print(count_vectorized_df.iloc[:3, 400:403].to_markdown())

# Question 3: Token counts of fourth tweet (index 3)
print("\nQuestion 3 - Token counts of fourth tweet:")
print(count_vectorized_df.iloc[3])

# Question 4: 15 most used words
print("\nQuestion 4 - 15 most used tokenized words:")
word_counts = count_vectorized_df.sum(axis=0).sort_values(ascending=False)
print(word_counts.head(15))

# Question 5: Add label column
count_vectorized_df['label'] = labels
print("\nQuestion 5 - DataFrame with labels (rows 350-353, last 2 columns):")
print(count_vectorized_df.iloc[350:354, -2:].to_markdown())
