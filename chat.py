import random
import json
import requests
from bs4 import BeautifulSoup
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'rb') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sampurnakart"
    
    
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I don't get what you asked \n Here are some of the link that can help you \n <li><a href='https://sampurnakart.in/repair/65c3196b0f4ad1db82b00eca' style='color: black; font-weight: bold;'>Smartphone Brands,</a></li>\n<li><a href='https://sampurnakart.in/repair/65c5e416372328de86ce2718' style='color: black; font-weight: bold;'>Laptop Brands,</a></li>\n<li><a href='https://sampurnakart.in/repair/65ded4742c80e64a6a685273' style='color: black; font-weight: bold;'>Tablet Brands,</a></li>\n<li><a href='https://sampurnakart.in/repair/65ded4df2c80e64a6a685291' style='color: black; font-weight: bold;'>Smart Watches,</a></li>\n<li><a href='https://sampurnakart.in/repair/65e963052c80e64a6a69f820' style='color: black; font-weight: bold;'>Heaphones Brand,</a></li>\n<li><a href='https://sampurnakart.in/repair/65f8066024e42d3794d993d9' style='color: black; font-weight: bold;'>Speakers</a></li>\n<li><a href='https://sampurnakart.in/repair/660285a64e982ea3bc6b291e' style='color: black; font-weight: bold;'>LED TV,</a></li>\n<li><a href='https://sampurnakart.in/repair/672b19e9a0958b6cb551d3a9' style='color: black; font-weight: bold;'>CC TV Services,</a></li>\n<li><a href='https://sampurnakart.in/repair/674ac7e466f51ddcb421994c' style='color: black; font-weight: bold;'>Car Services.</a></li>"

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

