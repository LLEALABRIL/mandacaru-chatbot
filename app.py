from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from joblib import load


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = load('trained_model_SVM.joblib')


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):

    # Let's chat for 5 lines
    for step in range(5):

        lemmatized = normalize(text)
        
        input_ids = tokenizer.encode(' '.join(lemmatized), return_tensors='pt')

        # Predict using the pre-trained model
        output = input_ids
        
        # Decode the output using the tokenizer (assuming it's a Hugging Face Transformers tokenizer)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
def normalize(text):
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    lowercase_words = [word.lower() for word in tokens if word.isalpha()]
    no_stopwords = [word for word in lowercase_words if word not in stopwords]
    lemma = nltk.WordNetLemmatizer()
    lemmatized = [lemma.lemmatize(word) for word in no_stopwords]
    return lemmatized
if __name__ == '__main__':
    app.run()


