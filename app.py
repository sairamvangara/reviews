from flask import Flask, render_template, redirect, request, session, url_for
#from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
import pandas as pd
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask_sqlalchemy import SQLAlchemy
data = pd.read_csv("all products_reviews.csv")
data1 = 'model.pkl'
model = pickle.load(open(data1, 'rb'))
vect = pickle.load(open('transform.pkl', 'rb'))
tfidf = pickle.load(open('transform1.pkl', 'rb'))

app = Flask(__name__)
app.secret_key = 'DSRAP'  # Change this to a random secret key for security

mydb = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Devika@19',
    port='3306',
    database='userlogin'
)

@app.route('/signup', methods=['GET', 'POST'])
def signup_post():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        #hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Insert user into the database
        mycursor = mydb.cursor()
        sql = "INSERT INTO user (username, password) VALUES (%s, %s)"
        val = (username, password)
        mycursor.execute(sql, val)
        mydb.commit()
        #mycursor.execute("SELECT * FROM user")

        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/')
def login():
    return render_template('login.html')
@app.route('/login', methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password']

    # Retrieve user from the database
    mycursor = mydb.cursor()
    sql = "SELECT id, username, password FROM user WHERE username = %s"
    val = (username,)
    mycursor.execute(sql, val)
    user = mycursor.fetchone()
    #if user and check_password_hash(user[2], password):
    if user and user[2] == password:
        session['username'] = user[1]
        return redirect('/product_review')
    else:
        return "Invalid login credentials. Please try again."


@app.route('/product_review')
def product_review():
    if 'username' in session:
        return render_template('product_review.html', username=session['username'])
    else:
        #return redirect('/predict')
        return redirect(url_for('/predict'))


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    product_name = request.form.get("product_name")
    print(product_name)
      # Import your SentimentAnalyzer class from model.py
      # Initialize your Flask app and other necessary configurations
      #product_name = "Mysore-Sandal-Soaps-Pack-Bars"
      # Filter reviews for the user-given product_name
    selected_reviews = [review["review"] for index, review in data.iterrows() if
                        review["name"] == product_name]

    if not selected_reviews:
        print("Error: Product not found.")
    else:
        # Sentiment Analysis
        nltk.download("vader_lexicon")
        sid = SentimentIntensityAnalyzer()
        positive_reviews = 0
        negative_reviews = 0

        # Calculate Positive and Negative Percentages
        for review in selected_reviews:
            sentiment_score = sid.polarity_scores(review)
            if sentiment_score["compound"] >= 0.05:
                positive_reviews += 1
            elif sentiment_score["compound"] <= -0.05:
                negative_reviews += 1

        total_reviews = len(selected_reviews)
        positive_percentage = (positive_reviews / total_reviews) * 100
        negative_percentage = (negative_reviews / total_reviews) * 100
        comments = selected_reviews
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(comments)
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        average_cosine_similarity = cosine_similarities.mean(axis=1)
        sorted_comments_indices = average_cosine_similarity.argsort()[::-1]
        top_comments_indices = sorted_comments_indices[:5]
        summary = '  '.join([comments[idx] for idx in top_comments_indices])
        max_summary_length = 300  # Set the maximum number of characters for the summary
        limited_summary = summary[:max_summary_length]
        words = limited_summary.split()
        lines=[]
        words_per_line = 300
        for i in range(0, len(words), words_per_line):
            # Join words to form a line of text
            line = '  '.join(words[i:i + words_per_line])
            lines.append(line)
            print(lines)
            #lines=line
            return render_template('product_review.html',
                                   product_name=product_name,summary_lines=line,
                                   prog=f'Positive: {positive_percentage:.2f}%,'
                                        f' Negative: {negative_percentage:.2f}%,'
                                        f'total_reviews: {total_reviews:.2f}')

        if len(words) > max_summary_length:
            lines.append(" ")





if __name__ == '__main__':
    app.run(debug=True)
