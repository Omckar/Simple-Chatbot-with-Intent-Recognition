import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam


def simple_tokenizer(text):
    return text.lower().split()

with open('intents.json', 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['.', '?', '!', ',', ';']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = simple_tokenizer(pattern)
        words.extend(word_list)

        tag_key = 'tag' if 'tag' in intent else 'tags'
        documents.append((word_list, intent[tag_key]))

        if intent[tag_key] not in classes:
            classes.append(intent[tag_key])

words = [word.lower() for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [word.lower() for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

max_sequence_length = max(len(item[0]) for item in training)
for i, item in enumerate(training):
    training[i][0] += [0] * (max_sequence_length - len(item[0]))

random.shuffle(training)

# Separate features (train_x) and labels (train_y)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
print("Done")
