from flask import Flask, render_template, request
from flask_frozen import Freezer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
import nltk
nltk.download('stopwords')

url = "https://raw.githubusercontent.com/EruditeStranger/Music_Rec/main/dataset_of_songs.csv"
data = pd.read_csv(url)

# Using requests
import requests

url = "https://raw.githubusercontent.com/EruditeStranger/Music_Rec/main/dataset_of_songs.csv"
response = requests.get(url)

if response.status_code == 200:
  # Process the downloaded data (e.g., save to a temporary file)
  with open("data.csv", "wb") as f:
    f.write(response.content)
else:
  print("Error downloading data:", response.status_code)

df = pd.read_csv("data.csv", low_memory=False)
df = df.drop_duplicates(subset="Song Name")

# Combine relevant columns into a single text column
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

# Initialize Flask app
app = Flask(__name__)
freezer = Freezer(app)  # Create a Frozen instance

# Define route for the recommendation page
@app.route("/")
def recommend_songs():
  return render_template("index.html")

# Define route for handling user input and recommendations
@app.route("/recommend", methods=["POST"])
def recommend():
  song_name = request.form["song_name"]

  if song_name in df["Song Name"].tolist():
    song_index = df["Song Name"].tolist().index(song_name)
    nearest_neighbors = knn.kneighbors(vectorized[song_index].reshape(1, -1))[1][0]
    recommendations = [df["Song Name"].iloc[neighbor] for neighbor in nearest_neighbors if neighbor != song_index]
    if not recommendations:
      message = "No similar songs found."
    else:
      message = "You should check out these songs: " + ", ".join(recommendations)
  else:
    message = "Song not found in database."

  return render_template("results.html", message=message)

# Build command for generating static files (run this manually)
@app.cli.command
def build():
  freezer.freeze()

if __name__ == "__main__":
  app.run(debug=True)
