import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

import nltk
nltk.download('stopwords')

# Read the data
df = pd.read_csv("data.csv", low_memory=False)

# Remove duplicates
df = df.drop_duplicates(subset="Song Name")

# Drop Null values (optional, adjust based on your data quality)
df = df.dropna(axis=0)

# Combine relevant columns into a single text column
# Check and potentially convert data types to string (if appropriate)
if not pd.api.types.is_string_dtype(df["Song Name"]):
  try:
    df["Song Name"] = df["Song Name"].astype(str)
  except (ValueError, TypeError):
    print("Error: Unable to convert 'Song Name' column to string. Please check data for compatibility.")

if not pd.api.types.is_string_dtype(df["Artist Name"]):
  try:
    df["Artist Name"] = df["Artist Name"].astype(str)
  except (ValueError, TypeError):
    print("Error: Unable to convert 'Artist Name' column to string. Please check data for compatibility.")

# Proceed with combining text data
df["text_data"] = df[["Song Name", "Artist Name"]].apply(lambda x: " ".join(x), axis=1)


# (Optional) Preprocess text data (e.g., remove stop words, stemming/lemmatization)
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
df['text_data'] = df['text_data'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# TF-IDF vectorization with weighting
vectorizer = TfidfVectorizer(use_idf=True)
vectorized = vectorizer.fit_transform(df["text_data"])

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(vectorized, df["Song Name"])

# Streamlit App
st.title("Song Recommendation System")
st.write("Enter a song name and find similar songs based on title and artist")

# User input for song name
song_name = st.text_input("Enter song name:")

# Recommendation function
def recommend_songs(song_name):
  if song_name in df["Song Name"].tolist():
    song_index = df["Song Name"].tolist().index(song_name)
    nearest_neighbors = knn.kneighbors(vectorized[song_index].reshape(1, -1))[1][0]
    recommendations = [df["Song Name"].iloc[neighbor] for neighbor in nearest_neighbors if neighbor != song_index]
    if not recommendations:
      return "No similar songs found."
    else:
      return recommendations
  else:
    return "Song not found in database."

# Display recommendations
if st.button("Recommend"):
  recommendations = recommend_songs(song_name)
  if isinstance(recommendations, list):
    st.subheader("Recommended Songs:")
    for song in recommendations:
      st.write(song)
  else:
    st.write(recommendations)
