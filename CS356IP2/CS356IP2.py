
# IP2_Airline_Sentiment_Theme_Extraction.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Step 1: Load the dataset
df = pd.read_csv("Tweets.csv")  # Make sure to place the dataset in the same folder as this script
print("Dataset loaded:", df.shape)

# Step 2: Clean and preprocess text data
df = df.dropna(subset=['text'])  # Drop any rows where text is NaN
documents = df['text'].values

# Step 3: Convert to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(documents)

# Step 4: Use NMF to extract themes
n_topics = 5
nmf_model = NMF(n_components=n_topics, random_state=42)
nmf_features = nmf_model.fit_transform(tfidf)

# Step 5: Identify the dominant topic/theme for each tweet
theme_labels = nmf_features.argmax(axis=1)
df['theme'] = theme_labels

# Optional: Show top words for each theme
print("\nTop words per theme:")
for i, topic in enumerate(nmf_model.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [tfidf_vectorizer.get_feature_names_out()[j] for j in top_words_idx]
    print(f"Theme {i}: {', '.join(top_words)}")

# Step 6: Save to a new CSV with the 'theme' column added
output_filename = "Airline_Sentiment_Themes.csv"
df.to_csv(output_filename, index=False)
print(f"\nThemes saved to {output_filename}")
