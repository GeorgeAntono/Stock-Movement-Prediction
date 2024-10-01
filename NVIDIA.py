#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# In[2]:


# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# In[3]:


# Load the datasets
news_df = pd.read_csv('./data/us_equities_news_dataset.csv')
stock_df = pd.read_csv('./data/NVDA.csv')

# In[4]:


'''

This is the initial document filtering but the number of articles is too high to process in my laptop.

# Convert the date columns to datetime format for matching
news_df['Date'] = pd.to_datetime(news_df['release_date'])
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# Merge news data with stock prices based on publication date
merged_df = pd.merge(news_df, stock_df, on='Date', how='inner')

# Filter to keep only articles that have matching stock data
nvidia_df = merged_df[['content', 'Open', 'Close', 'Date']]

# Label the target variable based on the opening and closing prices
nvidia_df['target'] = np.where(nvidia_df['Open'] > nvidia_df['Close'], 0, 1)

# Display the first few rows to verify the merging and labeling
print("\nFiltered and Labeled Data:")
print(nvidia_df.head())

'''

# In[5]:


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
nvidia_df = merged_df[['content', 'Open', 'Close', 'Date']]

# Label the target variable based on the opening and closing prices
nvidia_df['target'] = np.where(nvidia_df['Open'] > nvidia_df['Close'], 0, 1)

# Display the first few rows to verify the merging and labeling
print("\nFiltered and Labeled Data:")
print(nvidia_df.head())

# In[6]:


# Check for duplicate documents based on the 'content' column
duplicate_docs = nvidia_df[nvidia_df['content'].duplicated(keep=False)]

# Display the duplicate documents (if any)
print(f"Number of duplicate documents found: {duplicate_docs.shape[0]}")
if not duplicate_docs.empty:
    print("Duplicate Documents:")
    print(duplicate_docs[['content']])

# Remove duplicate documents, keeping the first occurrence
nvidia_df = nvidia_df.drop_duplicates(subset='content', keep='first').reset_index(drop=True)

# Display the updated DataFrame
print(f"Number of documents after removing duplicates: {nvidia_df.shape[0]}")

# In[7]:


# Basic descriptive statistics for the news dataset
num_articles = nvidia_df.shape[0]
average_words_per_article = nvidia_df['content'].apply(lambda x: len(str(x).split())).mean()
print(f'The number of articles before filtering is: {news_df.shape[0]}')
print(f"The number of articles after filtering is: {num_articles}")
print(f"The average amount of words per article is: {average_words_per_article}")

# In[8]:


# Initialize the stemmer
stemmer = PorterStemmer()

# In[9]:


# Tokenize content for word frequency analysis
nvidia_df.loc[:, 'text_length'] = nvidia_df['content'].apply(lambda x: len(str(x).split()))

# In[10]:


# Tokenize content for word frequency analysis
nvidia_df.loc[:, 'processed_text'] = nvidia_df['content'].apply(lambda x: word_tokenize(str(x).lower()))

# In[11]:


# Remove stopwords and punctuation for better NLP insights
stop_words = set(stopwords.words('english'))
# Remove stopwords, punctuation, and apply stemming
nvidia_df.loc[:, 'filtered_text'] = nvidia_df['processed_text'].apply(
    lambda words: [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
)

# In[12]:


# Join the filtered words back into strings for TF-IDF
nvidia_df.loc[:, 'filtered_text_str'] = nvidia_df['filtered_text'].apply(lambda x: ' '.join(x))

# In[13]:


# Frequency analysis of the most common words
all_words = [word for content in nvidia_df['filtered_text'] for word in content]
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(50)

# Display the most common words in the news dataset
print("\nTop 50 most common words in the news articles (after stop word removal):")
print(most_common_words)

# In[14]:


# TF-IDF Representation of Documents using the processed and filtered text
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Reduced max features
tfidf_matrix = vectorizer.fit_transform(nvidia_df['filtered_text_str'])

# In[15]:


# Train a linear classifier (e.g., Logistic Regression) on the TF-IDF matrix
clf = LogisticRegression(max_iter=1000)
clf.fit(tfidf_matrix, nvidia_df['target'])


# In[16]:


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

# In[17]:


# Visualize the top features for Class 0 (Stock Price Down)
plt.figure(figsize=(12, 6))
plt.barh([vectorizer.get_feature_names_out()[i] for i in top_features_class_0], coef[top_features_class_0],
         color='blue')
plt.title('Top 20 Words Indicative of Class 0 (Stock Price Down)')
plt.xlabel('Coefficient Value')
plt.gca().invert_yaxis()
plt.show()

# In[18]:


# Visualize the top features for Class 1 (Stock Price Up)
plt.figure(figsize=(12, 6))
plt.barh([vectorizer.get_feature_names_out()[i] for i in top_features_class_1], coef[top_features_class_1],
         color='green')
plt.title('Top 20 Words Indicative of Class 1 (Stock Price Up)')
plt.xlabel('Coefficient Value')
plt.gca().invert_yaxis()
plt.show()

# In[19]:


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

# In[20]:


# Compute cosine similarity matrix, which measures the similarity between documents

similarity_matrix = cosine_similarity(tfidf_matrix)

# Convert similarity matrix to a DataFrame for easier handling
similarity_df = pd.DataFrame(similarity_matrix)

# In[21]:


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

# In[22]:


# Find indices of the most and least similar documents (excluding self-similarity)
np.fill_diagonal(similarity_matrix, 0.0001)

# In[23]:


# Find the maximum value in the similarity matrix
max_similarity = np.max(similarity_matrix)
print(f"Maximum similarity value (excluding self-similarity): {max_similarity}")

# Find the minimum value in the similarity matrix
min_similarity = np.min(similarity_matrix)
print(f"Minimum similarity value (excluding self-similarity): {min_similarity}")

# In[24]:


# Create a heatmap of the similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap='viridis', cbar=True)
plt.title('Document Similarity Heatmap')
plt.xlabel('Documents')
plt.ylabel('Documents')
plt.show()

# https://datascience.stackexchange.com/questions/101862/cosine-similarity-between-sentence-embeddings-is-always-positive
#
#

# In[25]:


most_similar_indices = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
least_similar_indices = np.unravel_index(np.argmin(similarity_matrix, axis=None), similarity_matrix.shape)

# In[26]:


# Inspect the most similar documents
similar_doc_1 = nvidia_df.iloc[most_similar_indices[0]]
similar_doc_2 = nvidia_df.iloc[most_similar_indices[1]]

print("\nMost Similar Documents:\n")
print("Document 1:")
print(similar_doc_1['content'])
print("\nDocument 2:")
print(similar_doc_2['content'])

# In[27]:


# Inspect the least similar documents
dissimilar_doc_1 = nvidia_df.iloc[least_similar_indices[0]]
dissimilar_doc_2 = nvidia_df.iloc[least_similar_indices[1]]

print("\n\nMost Dissimilar Documents:\n")
print("Document 1:")
print(dissimilar_doc_1['content'])
print("\nDocument 2:")
print(dissimilar_doc_2['content'])

# In[ ]:


# Prepare the text data for Word2Vec
sentences = nvidia_df['filtered_text'].tolist()

# ## Word2Vec Model Training using CBOW

# In[28]:


# Initializing Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)  # sg=0 for CBOW

# In[29]:


# Training the model
word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)

# Saving the model for future use
word2vec_model.save("word2vec_model.bin")

# Getting word vectors for a particular word
word_vector = word2vec_model.wv['nvidia']


# Converting a document to an embedding by averaging word vectors
def get_document_embedding(doc):
    return np.mean([word2vec_model.wv[word] for word in doc if word in word2vec_model.wv], axis=0)


# Applying to the dataset
nvidia_df['doc_embedding'] = nvidia_df['filtered_text'].apply(get_document_embedding)


# In[ ]:


# Assuming 'model' is a trained Word2Vec model and 'tfidf' is the TfidfVectorizer
def document_embedding_tfidf(model, document, tfidf, feature_names, vector_size=100):
    """
    Compute a TF-IDF weighted document embedding by averaging the Word2Vec embeddings of words in the document.

    Parameters:
    - model: Trained Word2Vec model
    - document: List of tokenized words
    - tfidf: TfidfVectorizer object used for the document
    - feature_names: The list of features (terms) from the TF-IDF vectorizer
    - vector_size: Size of the word embeddings in the Word2Vec model

    Returns:
    - doc_embedding: The TF-IDF weighted document embedding as a numpy array
    """
    word_vectors = []
    weights = []

    # Extract the TF-IDF scores for the document
    tfidf_scores = tfidf.transform([" ".join(document)])
    tfidf_scores = tfidf_scores.toarray().flatten()

    for word in document:
        if word in model.wv.index_to_key and word in feature_names:
            # Get the word embedding
            word_vector = model.wv[word]
            word_index = feature_names.index(word)  # Get the index of the word in the TF-IDF feature list

            # Get the TF-IDF weight for this word
            tfidf_weight = tfidf_scores[word_index]

            # Collect the word vector and weight
            word_vectors.append(word_vector)
            weights.append(tfidf_weight)

    if len(word_vectors) == 0:
        # Return a zero vector if no words from the document are in the Word2Vec model
        return np.zeros(vector_size)

    # Convert lists to arrays
    word_vectors = np.array(word_vectors)
    weights = np.array(weights)

    # Compute the weighted average of word vectors
    doc_embedding = np.average(word_vectors, axis=0, weights=weights)

    return doc_embedding

# Assuming `vectorizer` is your trained TfidfVectorizer
tfidf_feature_names = vectorizer.get_feature_names_out()

# Apply the function to all documents in your filtered dataframe
nvidia_df['doc_embedding'] = nvidia_df['filtered_text'].apply(
    lambda doc: document_embedding_tfidf(word2vec_model, doc, vectorizer, tfidf_feature_names)
)
doc_embedding = document_embedding_tfidf(word2vec_model, document_words, tfidf, tfidf_feature_names)

# In[33]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Feature Matrix and Target Variable
X = tfidf_matrix
y = nvidia_df['target']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)

# Optional: Apply scaling to improve model performance (though it's not always necessary for TF-IDF)
# Initialize the scaler and create a pipeline with scaling and the classifier
scaler = StandardScaler(with_mean=False)  # with_mean=False because TF-IDF produces sparse matrices

# Example with Logistic Regression
lr_pipeline = make_pipeline(scaler, LogisticRegression(max_iter=1000))

# Train the model
lr_pipeline.fit(X_train, y_train)

# Evaluate on test data
test_score = lr_pipeline.score(X_test, y_test)

print(f"Test accuracy of Logistic Regression: {test_score:.2f}")

# Optionally, use cross-validation to validate the model across multiple folds
cross_val_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy: {np.mean(cross_val_scores):.2f} ± {np.std(cross_val_scores):.2f}")


# In[34]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

#

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# In[35]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Predictions for both models
nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Evaluation metrics
accuracy_nb = accuracy_score(y_test, nb_pred)
accuracy_lr = accuracy_score(y_test, lr_pred)

f1_nb = f1_score(y_test, nb_pred)
f1_lr = f1_score(y_test, lr_pred)

print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}, F1 Score: {f1_nb:.2f}")
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}, F1 Score: {f1_lr:.2f}")

# Confusion Matrix
cm_nb = confusion_matrix(y_test, nb_pred)
cm_lr = confusion_matrix(y_test, lr_pred)

print("\nNaive Bayes Confusion Matrix:")
print(cm_nb)

print("\nLogistic Regression Confusion Matrix:")
print(cm_lr)

# In[46]:


## Word2Vec Model Training using Skip-Gram


# In[41]:


# Initializing Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=200, window=5, min_count=1, workers=4, sg=1)  # sg=1 for Skip-Gram

# In[42]:


# Training the model
word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)

# Saving the model for future use
word2vec_model.save("word2vec_model_skipgram.bin")

# Getting word vectors for a particular word
word_vector_2 = word2vec_model.wv['nvidia']


# Converting a document to an embedding by averaging word vectors
def get_document_embedding(doc):
    return np.mean([word2vec_model.wv[word] for word in doc if word in word2vec_model.wv], axis=0)


# Applying to the dataset
nvidia_df['doc_embedding'] = nvidia_df['filtered_text'].apply(get_document_embedding)

# In[43]:


from sklearn.model_selection import train_test_split

# First step is to split the data into training and test sets
X = tfidf_matrix
y = nvidia_df['target']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)

# In[44]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

#

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# In[45]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Predictions for both models
nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Evaluation metrics
accuracy_nb = accuracy_score(y_test, nb_pred)
accuracy_lr = accuracy_score(y_test, lr_pred)

f1_nb = f1_score(y_test, nb_pred)
f1_lr = f1_score(y_test, lr_pred)

print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}, F1 Score: {f1_nb:.2f}")
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}, F1 Score: {f1_lr:.2f}")

# Confusion Matrix
cm_nb = confusion_matrix(y_test, nb_pred)
cm_lr = confusion_matrix(y_test, lr_pred)

print("\nNaive Bayes Confusion Matrix:")
print(cm_nb)

print("\nLogistic Regression Confusion Matrix:")
print(cm_lr)

# In[ ]:




