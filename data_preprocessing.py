#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import string
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# In[5]:


# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# In[6]:


# Load the datasets
news_df = pd.read_csv('./data/us_equities_news_dataset.csv')
stock_df = pd.read_csv('./data/NVDA.csv')

# In[7]:


'''

This is the initial document filtering but the number of articles is too high to process in my laptop.

# Convert the date columns to datetime format for matching
news_df['Date'] = pd.to_datetime(news_df['release_date'])
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# Merge news data with stock prices based on publication date
merged_df = pd.merge(news_df, stock_df, on='Date', how='inner')

# Filter to keep only articles that have matching stock data
filtered_df = merged_df[['content', 'Open', 'Close', 'Date']]

# Label the target variable based on the opening and closing prices
filtered_df['target'] = np.where(filtered_df['Open'] > filtered_df['Close'], 0, 1)

# Display the first few rows to verify the merging and labeling
print("\nFiltered and Labeled Data:")
print(filtered_df.head())

'''

# In[8]:


# Sample keywords related to NVIDIA and associated companies
nvidia_keywords = [
    'NVDA', 'NVIDIA', 'GPU', 'GRAPHICS']

'''

More keywords can be added to the list to improve the filtering process. But adding all those increases the document number significantly making processing slower and creating memory issues that are unsolvable by my mere laptop. 

nvidia_keywords = [
    'NVDA', 'NVIDIA', 'NIO', 'UBER', 'AMZN', 'AMAZON', 'TESLA', 'AI', 'GPU', 'GRAPHICS',
    'CHIP', 'SEMICONDUCTOR', 'AUTONOMOUS', 'DRIVING', 'DEEP LEARNING', 'MACHINE LEARNING'
]

'''
# Compile a regex pattern from the keywords list
nvidia_pattern = '|'.join(nvidia_keywords)  # Combines the keywords into a regex pattern

# Filter articles where the content or ticker column contains any of the keywords
nvidia_related_articles = news_df[
    news_df['content'].str.contains(nvidia_pattern, case=False, na=False) |
    news_df['ticker'].str.contains(nvidia_pattern, case=False, na=False)
    ]

# Display the count of NVIDIA-related articles
print(f"\nTotal NVIDIA-related articles found: {nvidia_related_articles.shape[0]}")

# Convert the date columns to datetime format for matching
nvidia_related_articles['Date'] = pd.to_datetime(nvidia_related_articles['release_date'])
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# Merge filtered news data with stock prices based on publication date
merged_df = pd.merge(nvidia_related_articles, stock_df, on='Date', how='inner')

# Filter to keep only articles that have matching stock data
filtered_df = merged_df[['content', 'Open', 'Close', 'Date']]

# Label the target variable based on the opening and closing prices
filtered_df['target'] = np.where(filtered_df['Open'] > filtered_df['Close'], 0, 1)

# Display the first few rows to verify the merging and labeling
print("\nFiltered and Labeled Data:")
print(filtered_df.head())

# In[9]:


# Check for duplicate documents based on the 'content' column
duplicate_docs = filtered_df[filtered_df['content'].duplicated(keep=False)]

# Display the duplicate documents (if any)
print(f"Number of duplicate documents found: {duplicate_docs.shape[0]}")
if not duplicate_docs.empty:
    print("Duplicate Documents:")
    print(duplicate_docs[['content']])

# Remove duplicate documents, keeping the first occurrence
filtered_df = filtered_df.drop_duplicates(subset='content', keep='first').reset_index(drop=True)

# Display the updated DataFrame
print(f"Number of documents after removing duplicates: {filtered_df.shape[0]}")

# In[10]:


# Basic descriptive statistics for the news dataset
num_articles = filtered_df.shape[0]
average_words_per_article = filtered_df['content'].apply(lambda x: len(str(x).split())).mean()
print(f'The number of articles before filtering is: {news_df.shape[0]}')
print(f"The number of articles after filtering is: {num_articles}")
print(f"The average amount of words per article is: {average_words_per_article}")

# In[11]:


# Initialize the stemmer
stemmer = PorterStemmer()

# In[12]:


# Tokenize content for word frequency analysis
filtered_df.loc[:, 'text_length'] = filtered_df['content'].apply(lambda x: len(str(x).split()))

# In[13]:


# Tokenize content for word frequency analysis
filtered_df.loc[:, 'processed_text'] = filtered_df['content'].apply(lambda x: word_tokenize(str(x).lower()))

# In[14]:


# Remove stopwords and punctuation for better NLP insights
stop_words = set(stopwords.words('english'))
# Remove stopwords, punctuation, and apply stemming
filtered_df.loc[:, 'filtered_text'] = filtered_df['processed_text'].apply(
    lambda words: [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
)

# In[15]:


# Join the filtered words back into strings for TF-IDF
filtered_df.loc[:, 'filtered_text_str'] = filtered_df['filtered_text'].apply(lambda x: ' '.join(x))

# In[16]:


# Frequency analysis of the most common words
all_words = [word for content in filtered_df['filtered_text'] for word in content]
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(50)

# Display the most common words in the news dataset
print("\nTop 50 most common words in the news articles (after stop word removal):")
print(most_common_words)

# In[17]:


# TF-IDF Representation of Documents using the processed and filtered text
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Reduced max features
tfidf_matrix = vectorizer.fit_transform(filtered_df['filtered_text_str'])

# In[18]:


# Train a linear classifier (e.g., Logistic Regression) on the TF-IDF matrix
clf = LogisticRegression(max_iter=1000)
clf.fit(tfidf_matrix, filtered_df['target'])


# In[19]:


def print_top_features(vectorizer, coef):
    """Prints features with the highest coefficient values for binary classification."""
    feature_names = vectorizer.get_feature_names_out()

    # Top 20 features with the most negative coefficients (indicative of Class 0)
    top_features_class_0 = np.argsort(coef)[:20]
    print("\nTop 20 Words Indicative of Class 0 (Stock Price Down):")
    print(" ".join(feature_names[j] for j in top_features_class_0))

    # Top 20 features with the most positive coefficients (indicative of Class 1)
    top_features_class_1 = np.argsort(coef)[-20:]
    print("\nTop 20 Words Indicative of Class 1 (Stock Price Up):")
    print(" ".join(feature_names[j] for j in top_features_class_1))

    return top_features_class_0, top_features_class_1


# Extract the coefficients from the classifier
coef = clf.coef_[0]

# Print the most informative features and get the indices for visualization
top_features_class_0, top_features_class_1 = print_top_features(vectorizer, coef)

# In[20]:


# Visualize the top features for Class 0 (Stock Price Down)
plt.figure(figsize=(12, 6))
plt.barh([vectorizer.get_feature_names_out()[i] for i in top_features_class_0], coef[top_features_class_0],
         color='blue')
plt.title('Top 20 Words Indicative of Class 0 (Stock Price Down)')
plt.xlabel('Coefficient Value')
plt.gca().invert_yaxis()
plt.show()

# In[21]:


# Visualize the top features for Class 1 (Stock Price Up)
plt.figure(figsize=(12, 6))
plt.barh([vectorizer.get_feature_names_out()[i] for i in top_features_class_1], coef[top_features_class_1],
         color='green')
plt.title('Top 20 Words Indicative of Class 1 (Stock Price Up)')
plt.xlabel('Coefficient Value')
plt.gca().invert_yaxis()
plt.show()

# In[22]:


# Check the shape of the tfidf_matrix
print(f"Shape of tfidf_matrix: {tfidf_matrix.shape}")

# Number of documents (rows) and terms (columns)
num_documents, num_features = tfidf_matrix.shape
print(f"Number of documents: {num_documents}")
print(f"Number of features (terms): {num_features}")

# Check the number of non-zero elements
non_zero_elements = tfidf_matrix.nnz
print(f"Number of non-zero elements: {non_zero_elements}")

# Calculate the total number of elements
total_elements = num_documents * num_features
print(f"Total number of elements: {total_elements}")

# Calculate the sparsity of the matrix
sparsity = (1 - (non_zero_elements / total_elements)) * 100
print(f"Sparsity of the tfidf_matrix: {sparsity:.2f}%")

# In[51]:


# Compute cosine similarity matrix, which measures the similarity between documents

similarity_matrix = cosine_similarity(tfidf_matrix)

# Convert similarity matrix to a DataFrame for easier handling
similarity_df = pd.DataFrame(similarity_matrix)

# In[52]:


# Check the shape of the similarity_matrix
print(f"Shape of similarity_matrix: {similarity_matrix.shape}")

# Number of documents (rows and columns since it's square)
num_documents = similarity_matrix.shape[0]
print(f"Number of documents: {num_documents}")

# Check the number of non-zero elements in the similarity matrix
non_zero_elements = np.count_nonzero(similarity_matrix)
print(f"Number of non-zero elements: {non_zero_elements}")

# Calculate the total number of elements in the similarity matrix
total_elements = num_documents * num_documents
print(f"Total number of elements: {total_elements}")

# Calculate the sparsity of the similarity matrix
sparsity = (1 - (non_zero_elements / total_elements)) * 100
print(f"Sparsity of the similarity_matrix: {sparsity:.2f}%")

# In[53]:


# Find indices of the most and least similar documents (excluding self-similarity)
np.fill_diagonal(similarity_matrix, 0.0001)

# In[55]:


# Find the maximum value in the similarity matrix
max_similarity = np.max(similarity_matrix)
print(f"Maximum similarity value (excluding self-similarity): {max_similarity}")

# Find the minimum value in the similarity matrix
min_similarity = np.min(similarity_matrix)
print(f"Minimum similarity value (excluding self-similarity): {min_similarity}")

# In[56]:


# Create a heatmap of the similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap='viridis', cbar=True)
plt.title('Document Similarity Heatmap')
plt.xlabel('Documents')
plt.ylabel('Documents')
plt.show()

# In[57]:


most_similar_indices = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
least_similar_indices = np.unravel_index(np.argmin(similarity_matrix, axis=None), similarity_matrix.shape)

# In[58]:


# Inspect the most similar documents 
similar_doc_1 = filtered_df.iloc[most_similar_indices[0]]
similar_doc_2 = filtered_df.iloc[most_similar_indices[1]]

print("\nMost Similar Documents:\n")
print("Document 1:")
print(similar_doc_1['content'])
print("\nDocument 2:")
print(similar_doc_2['content'])

# In[59]:


# Inspect the least similar documents
dissimilar_doc_1 = filtered_df.iloc[least_similar_indices[0]]
dissimilar_doc_2 = filtered_df.iloc[least_similar_indices[1]]

print("\n\nMost Dissimilar Documents:\n")
print("Document 1:")
print(dissimilar_doc_1['content'])
print("\nDocument 2:")
print(dissimilar_doc_2['content'])





