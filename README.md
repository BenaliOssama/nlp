# NLP: Text Preprocessing and Bag of Words

## Overview

This project covers Natural Language Processing fundamentals, specifically text preprocessing and bag-of-words representation. Machine learning models cannot work with raw text directly—it must be converted to numerical vectors. This project demonstrates the complete pipeline from raw text to a feature matrix suitable for model training.

## Project Structure

```
.
├── ex01/          # Exercise 1: Lowercase
├── ex02/          # Exercise 2: Punctuation removal
├── ex03/          # Exercise 3: Tokenization
├── ex04/          # Exercise 4: Stop word removal
├── ex05/          # Exercise 5: Stemming
├── ex06/          # Exercise 6: Text preprocessing function
├── ex07/          # Exercise 7: Bag of Words representation
├── resources/     # Data files (tweets_train.txt)
└── requirements.txt
```

## Setup

### Create Virtual Environment
```bash
python3 -m venv ex00
source ex00/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download NLTK Data
NLTK requires pre-trained models. These download automatically on first run:
- `punkt_tab`: Sentence and word tokenizer
- `stopwords`: Common English stop words

## Exercises

### Exercise 1: Lowercase
**Goal:** Convert text to lowercase for consistent processing.

**Key Concept:** "This", "THIS", and "this" should be treated as the same word.

**Implementation:** Use Pandas `.str.lower()` and `.str.upper()` on Series.

### Exercise 2: Punctuation Removal
**Goal:** Remove punctuation marks that don't carry semantic meaning.

**Key Concept:** "I love this!" and "I love this" are identical for sentiment analysis.

**Implementation:** Use `str.translate()` with `str.maketrans()` to delete punctuation characters.

### Exercise 3: Tokenization
**Goal:** Split text into tokens (sentences or words).

**Key Concept:** Before processing, text must be broken into individual units.

**Implementation:** 
- Sentence tokenization: `sent_tokenize()` from NLTK
- Word tokenization: `word_tokenize()` from NLTK

### Exercise 4: Stop Word Removal
**Goal:** Remove common words that don't carry information ("the", "is", "a", "and").

**Key Concept:** Stop words appear everywhere and add noise. Removing them lets the model focus on meaningful words.

**Implementation:** Filter tokens against NLTK's English stop words list.

### Exercise 5: Stemming
**Goal:** Reduce words to their root form.

**Key Concept:** "running", "runs", "ran" are variations of "run". Stemming treats them as one feature.

**Implementation:** Use `PorterStemmer` from NLTK to reduce inflections.

**Note:** Stemmed output may not be valid dictionary words (e.g., "presid" for "president").

### Exercise 6: Text Preprocessing Function
**Goal:** Create a reusable preprocessing pipeline combining all previous steps.

**Pipeline Order:**
1. Lowercase
2. Remove punctuation
3. Tokenize into words
4. Remove stop words
5. Stem tokens

**Implementation:** Single function that applies all transformations and returns a list of cleaned tokens.

### Exercise 7: Bag of Words Representation
**Goal:** Convert preprocessed tweets into a word-count matrix for machine learning.

**Key Concept:** Each row = a document (tweet), each column = a word, each cell = word count.

**Implementation:**
1. Load and parse labeled tweets from `resources/tweets_train.txt`
2. Use scikit-learn's `CountVectorizer` to build a sparse matrix (6588 tweets × 500 features)
3. Convert sparse matrix to Pandas DataFrame for readability
4. Map sentiment labels: positive=1, negative=-1, neutral=0
5. Add labels as final column

**Why Sparse Matrix?** 
Most tweets don't contain most words. A dense matrix would be 6588 × 500 with mostly zeros—wasteful. Sparse matrices store only non-zero values.

## Data Format

### Input: tweets_train.txt
```
positive, Gas by my house hit $3.39!!!! I'm going to Chapel Hill on Sat. :)
negative, Theo Walcott is still shit, watch Rafa and Johnny deal with him on Saturday.
neutral, some neutral tweet here
```

Format: `label, tweet_text` (comma-separated)

### Output: count_vectorized_df
A DataFrame with:
- Rows: Individual tweets (6588 total)
- Columns: Word counts (500 most frequent words) + label column
- Values: Integer counts (how many times each word appears)
- Data type: Sparse (efficient storage)

Example structure:
```
     and  boat  compute  ...  your  label
0      0     2        0  ...     0      1
1      0     0        1  ...     1     -1
2      1     0        0  ...     0      0
```

## Key Concepts

### Why Each Step Matters

**Lowercase:** Standardizes variations so the model sees one feature, not multiple.

**Punctuation Removal:** Punctuation doesn't affect sentiment classification in most cases.

**Tokenization:** Breaks continuous text into discrete units for analysis.

**Stop Word Removal:** Eliminates noise; focuses the model on discriminative words.

**Stemming:** Reduces feature dimensionality by grouping related word forms.

**Bag of Words:** Converts text to numerical format that ML algorithms require. Trades word order for simplicity and interpretability.

### Trade-offs

**Bag of Words Limitations:**
- Loses word order and context ("dog bites man" vs "man bites dog")
- Loses negation nuance ("not good" treated like positive words)
- All word positions treated equally

**When to Use BoW:**
- Simple classification tasks (sentiment analysis, spam detection)
- When computational efficiency matters
- When interpretability is important (can see which words drive predictions)

**Alternatives (not covered here):**
- TF-IDF: Weighs words by frequency and rarity
- Word embeddings (Word2Vec, GloVe): Preserve semantic relationships
- RNNs/LSTMs: Capture word order and context

## Running the Exercises

Each exercise is independent:
```bash
cd ex01
python3 ex01.py
```

Exercise 7 requires the data file:
```bash
cd ex07
python3 ex07.py
```

## Dependencies

See `requirements.txt` for complete list. Key packages:
- `nltk`: Tokenization, stop words, stemming
- `pandas`: Data manipulation
- `scikit-learn`: CountVectorizer, ML algorithms
- `jupyter`: Interactive notebooks (optional)

## References

- [Your Guide to Natural Language Processing (NLP)](https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [Gentle Introduction to Bag of Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)

## Learning Outcomes

After completing this project, you should understand:
- Why and how to preprocess text data
- The purpose of each preprocessing step
- How bag-of-words converts text to numerical features
- Limitations of bag-of-words representations
- How to use NLTK and scikit-learn for NLP tasks
- The data leakage principle: fit transformers on training data only
