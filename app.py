from flask import Flask, render_template, request
import nltk
import json
import requests
import os

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

#tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#model = load('trained_model_SVM.joblib')

url = 'https://mandacaru-sentiment-api.onrender.com/analise-sentimento'

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = get_Chat_response(msg)

    data = {
    'texto': input
    }

    json_data = json.dumps(data)
    
    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json_data)
        # Parse the JSON into a Python dictionary
    response_dict = json.loads(response.text)

    # Extract the first element of the list associated with the 'sentimento' key
    sentimento = response_dict["sentimento"][0]
    
    return sentimento.capitalize()

def get_Chat_response(text):
    lemmatized = normalize(text)
    normalized_msg = ' '.join(lemmatized)
    
    return normalized_msg
    
def normalize(text):
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    lowercase_words = [word.lower() for word in tokens if word.isalpha()]
    no_stopwords = [word for word in lowercase_words if word not in stopwords]
    lemma = nltk.WordNetLemmatizer()
    lemmatized = [lemma.lemmatize(word) for word in no_stopwords]
    return lemmatized

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    # Run the app, specifying 0.0.0.0 as the host to bind to
    app.run(host='0.0.0.0', port=port, debug=True)