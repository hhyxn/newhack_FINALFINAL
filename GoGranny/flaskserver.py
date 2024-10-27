from flask import Flask, render_template, request
import joblib
import model, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchtext
import string
from llama_index.llms.groq import Groq
import os
from llama_index.core.llms import ChatMessage

# Get the API key from environment variables
api_key = os.getenv('GROQ_API_KEY')

# Initialize the model with the API key
llm = Groq(model="llama3-70b-8192", api_key=api_key)

app = Flask(__name__)

def char_tokenizer(text):
    return list(text)

text_field = torchtext.data.Field(tokenize=char_tokenizer,  #do we need sos and eos? not sure, ignore for now
                                  include_lengths=True,
                                  batch_first = True)
# Define the list of characters for the vocabulary
characters = list(string.ascii_lowercase + string.digits + string.punctuation + ' ')

# Use the characters to build the vocabulary
text_field.build_vocab(characters)
mymodel = model.LSTM(94, 64,3)
mymodel.load_state_dict(torch.load('./pretrained_model.pth'))
mymodel.eval()


@app.route('/', defaults={ 'path': 'main' })
@app.route('/<path>', methods = ['GET'])
def index(path):
    try:
        return render_template(path + '.html')
    except:
        print(path + " does not exist")
        return ''

@app.route('/passdetect', methods = ['POST'])
def classify():
    pwd = request.form.get('password')
    #print(pwd)
    pwd_index = [text_field.vocab.stoi[char] for char in pwd]
    input = torch.tensor(pwd_index).unsqueeze(0)

    with torch.no_grad():
        output = mymodel(input, torch.tensor([len(pwd)]))
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1).item()

    if (output == 0):
        strength = "weak"
    elif(output == 1):
        strength = "moderate"
    else:
        strength = "strong"
    #print(strength)
    return render_template('passdetect.html', predict = strength)

@app.route('/passgenerate', methods = ['POST'])
def generate():
    a = request.form.get('q1')
    b = request.form.get('q2')
    c = request.form.get('q3')
    final = a+ " " + b + " " + c

    messages = [
        ChatMessage(
            role="system", content="You are a password generator. You will be given three key words and your only job is to generate a strong mixed up, hard to guess password containing the words and some random symbols but not too many. give a result in this format: password:"
        ),
        ChatMessage(role="user", content=final),
    ]
    resp = llm.chat(messages)
    password = str(resp)
    password = password[21:len(password)]

    #story
    messages = [
        ChatMessage(
            role="system", content="You are a very simple story a wholesome story a elder can understand generator which is 80 words(dont mention its 80 words ANYWHERE!!) for someone who needs to remember their passwords. You must use the 3 key words, even if you don't recognize the word as it might be a name."
        ),
        ChatMessage(role="user", content= "word 1 = " + a + " word 2 =" + b + " word 3 =" + c),
    ]
    resp = llm.chat(messages)
    story = str(resp)
    story = story[11:len(story)]
    
    return render_template('passgenerate.html', pwd = password, stry = story)


if __name__ == '__main__':
    app.run(port=3001, debug=True)