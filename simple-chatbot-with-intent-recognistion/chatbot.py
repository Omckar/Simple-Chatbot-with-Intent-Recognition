import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set TensorFlow logging level to suppress verbose messages
tf.get_logger().setLevel('ERROR')

def load_intents(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_data(file_path):
    return pickle.load(open(file_path, 'rb'))

intents = load_intents('intents.json')
words = load_data('words.pkl')
classes = load_data('classes.pkl')
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_data):
    tag = intents_list[0]['intent']
    list_of_intents = intents_data['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("GO! Bot is running")

while True:
    message = input("You: ")
    if message.lower() == 'quit':
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f"Bot: {res}")
