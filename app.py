from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/content.csv')

# Create a simple recommendation system
def recommend_content(user_preference):
    # Initialize CountVectorizer
    vectorizer = CountVectorizer()
    
    # Fit and transform the content genre
    genre_matrix = vectorizer.fit_transform(df['genre'])
    
    # Compute similarity between user preference and content genres
    user_vector = vectorizer.transform([user_preference])
    similarities = cosine_similarity(user_vector, genre_matrix).flatten()
    
    # Get the indices of the most similar contents
    indices = similarities.argsort()[-5:][::-1]
    recommended_contents = df.iloc[indices]
    
    return recommended_contents

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        preference = request.form['preference']
        recommendations = recommend_content(preference)
        return render_template('recommendations.html', recommendations=recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
