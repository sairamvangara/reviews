from flask import Flask, request, render_template
import pandas as pd
import pickle

data = pd.read_csv("all products_reviews.csv")

data1 = 'model.pkl'
model = pickle.load(open(data1, 'rb'))
vect = pickle.load(open('transform.pkl', 'rb'))
tfidf = pickle.load(open('transform1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('i.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    comment = [x for x in request.form.values()]
    print(comment)
    x = data.iloc[:, :-1].values
    x = vect.transform(comment)
    x_tfidf = tfidf.transform(x)

    o = model.predict(x_tfidf)
    print(o)

    if o[0] == 0:
        return render_template('i.html', prog=' not upto the mark🤬')
    elif o[0] == 1:
        return render_template('i.html',
                               prog='As per prediction most of the customers feels product is  upto the mark,they are happy😒')
    elif o[0] == 2:
        return render_template('i.html', prog=' product is just satisfying😑')
