from llama_index.llms.groq import Groq
import os
from llama_index.core.llms import ChatMessage

# Get the API key from environment variables
api_key = os.getenv('GROQ_API_KEY')

# Initialize the model with the API key
llm = Groq(model="llama3-70b-8192", api_key=api_key)

a="adeel"
b="eating"
c="3453"
final = a+ " " + b + " " + c

messages = [
    ChatMessage(
        role="system", content="You are a password generator. You will be given three key words and your only job is to generate a strong mixed up, hard to guess password containing the words and some random symbols but not too many. give a result in this format: password:"
    ),
    ChatMessage(role="user", content=final),
]
resp = llm.chat(messages)

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
