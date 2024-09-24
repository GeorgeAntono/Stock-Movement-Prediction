import pandas as pd
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import chi2

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the datasets
news_df = pd.read_csv('./data/us_equities_news_dataset.csv')
stock_df = pd.read_csv('./data/NVDA.csv')

# Convert the date columns to datetime format for matching
news_df['Date'] = pd.to_datetime(news_df['release_date'])
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# Merge news data with stock prices based on publication date
merged_df = pd.merge(news_df, stock_df, on='Date', how='inner')

# Filter to keep only articles that have matching stock data
filtered_df = merged_df[['content', 'Open', 'Close', 'Date']]

# Label the target variable based on the opening and closing prices
filtered_df.loc[:, 'target'] = np.where(filtered_df.loc[:, 'Open'] > filtered_df.loc[:, 'Close'], 0, 1)

# Display the first few rows to verify the merging and labeling
print("\nFiltered and Labeled Data:")
print(filtered_df.head())

# Basic descriptive statistics for the filtered dataset
num_articles = filtered_df.shape[0]
average_words_per_article = filtered_df['content'].apply(lambda x: len(str(x).split())).mean()
print(f"The number of articles is: {num_articles}")
print(f"The average number of words per article is: {average_words_per_article}")

# Text length analysis for filtered news articles
filtered_df.loc[:, 'text_length'] = filtered_df['content'].apply(lambda x: len(str(x).split()))
print(f"\nAverage text length of filtered news articles: {average_words_per_article:.2f} words")

# Tokenize content for word frequency analysis
filtered_df.loc[:, 'processed_text'] = filtered_df['content'].apply(lambda x: word_tokenize(str(x).lower()))

# Remove stopwords and punctuation for better NLP insights
stop_words = set(stopwords.words('english'))
filtered_df.loc[:, 'filtered_text'] = filtered_df['processed_text'].apply(
    lambda words: [word for word in words if word.isalpha() and word not in stop_words])

# Join the filtered words back into strings for TF-IDF
filtered_df.loc[:, 'filtered_text_str'] = filtered_df['filtered_text'].apply(lambda x: ' '.join(x))

# Frequency analysis of the most common words
all_words = [word for content in filtered_df['filtered_text'] for word in content]
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(50)

# Display the most common words in the filtered news dataset
print("\nTop 50 most common words in the filtered news articles (after stop word removal):")
print(most_common_words)

# Display general statistics for the filtered dataset
overview_stats = {
    'Number of Articles': num_articles,
    'Average Words per Article': average_words_per_article,
    'Number of Unique Words': len(set(all_words)),
    'Lexical Richness': len(set(all_words)) / len(all_words) if len(all_words) > 0 else 0
}

print("\nOverview Statistics for the Filtered Dataset:")
for stat, value in overview_stats.items():
    print(f"{stat}: {value}")

# Plot the frequency of the top 50 words
plt.figure(figsize=(10, 6))
plt.bar(*zip(*most_common_words))
plt.title("Top 50 Most Common Words in Filtered News Articles")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# Distribution of article lengths in the filtered dataset
plt.figure(figsize=(10, 6))
plt.hist(filtered_df['text_length'], bins=50, color='skyblue')
plt.title('Distribution of Article Lengths in Filtered News Articles')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# TF-IDF Representation of Documents using the processed and filtered text
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_matrix = vectorizer.fit_transform(filtered_df['filtered_text_str'])

# Get feature names (words) from the TF-IDF vectorizer
feature_names = vectorizer.get_feature_names_out()

# Apply Chi-Square test to find words most indicative of each class
chi2_scores, p_values = chi2(tfidf_matrix, filtered_df['target'])

# Create a DataFrame of words and their Chi-Square scores
chi2_df = pd.DataFrame({'Word': feature_names, 'Chi2 Score': chi2_scores})

# Identify top indicative words for each class based on the average TF-IDF scores for each class
# Separate the TF-IDF matrix by class
tfidf_class_0 = tfidf_matrix[filtered_df['target'] == 0]
tfidf_class_1 = tfidf_matrix[filtered_df['target'] == 1]

# Calculate the mean TF-IDF scores for each word in each class
mean_scores_class_0 = np.mean(tfidf_class_0.toarray(), axis=0)
mean_scores_class_1 = np.mean(tfidf_class_1.toarray(), axis=0)

# Create DataFrames for each class's scores
class_0_df = pd.DataFrame({'Word': feature_names, 'Mean TF-IDF': mean_scores_class_0})
class_1_df = pd.DataFrame({'Word': feature_names, 'Mean TF-IDF': mean_scores_class_1})

# Identify the top 20 words most indicative of each class
top_words_class_0 = class_0_df.nlargest(20, 'Mean TF-IDF')['Word'].tolist()
top_words_class_1 = class_1_df.nlargest(20, 'Mean TF-IDF')['Word'].tolist()

# Display the top 20 words indicative of each class
print("\nTop 20 Words Indicative of Class 0:")
print(top_words_class_0)

print("\nTop 20 Words Indicative of Class 1:")
print(top_words_class_1)


'''

# TF-IDF Representation of Documents using the processed and filtered text
vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
tfidf_matrix = vectorizer.fit_transform(news_df['filtered_text_str'])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Convert similarity matrix to a DataFrame for easier handling
similarity_df = pd.DataFrame(similarity_matrix)

# Find indices of the most and least similar documents (excluding self-similarity)
np.fill_diagonal(similarity_matrix, 0)
most_similar_indices = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
least_similar_indices = np.unravel_index(np.argmin(similarity_matrix, axis=None), similarity_matrix.shape)

# Inspect the most similar documents
similar_doc_1 = news_df.iloc[most_similar_indices[0]]
similar_doc_2 = news_df.iloc[most_similar_indices[1]]

print("\nMost Similar Documents:\n")
print("Document 1:")
print(similar_doc_1['content'])
print("\nDocument 2:")
print(similar_doc_2['content'])

# Inspect the least similar documents
dissimilar_doc_1 = news_df.iloc[least_similar_indices[0]]
dissimilar_doc_2 = news_df.iloc[least_similar_indices[1]]

print("\n\nMost Dissimilar Documents:\n")
print("Document 1:")
print(dissimilar_doc_1['content'])
print("\nDocument 2:")
print(dissimilar_doc_2['content'])

'''