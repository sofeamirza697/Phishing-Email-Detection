from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# load and train the model
def load_trained_model():
    # load dataset
    df = pd.read_csv("phishing_email.csv")
    df = df.dropna() # data cleaning
    
    X = df["text_combined"]
    y = df["label"]

    # pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB())
    ])

    # training the model
    model.fit(X, y)
    return model

# global model instance
phish_model = load_trained_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('content', '')
    
    if not email_text:
        return jsonify({'error': 'No content'}), 400

    # 1 is Phishing, 0 is Safe based on your class distribution
    prediction = phish_model.predict([email_text])[0]
    result = "PHISHING DETECTED" if prediction == 1 else "SAFE EMAIL"
    
    return jsonify({'result': result, 'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)