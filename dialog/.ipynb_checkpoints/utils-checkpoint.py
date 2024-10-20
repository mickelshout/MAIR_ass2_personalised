import numpy as np
import pandas as pd
import Levenshtein
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
import pickle


df = pd.read_csv("../data/restaurant_info.csv")
food_types = list(np.unique(df.food)) + ['greek']
pricerange = np.unique(df.pricerange)
areas = list(np.unique(df.area.fillna('unknown')))
areas.remove("unknown")

"""
The folowing function retrives information given sentence from the user. It looks for food kind, pricerande and location.
"""

def find_preferences(sentence):
    sentence = sentence.lower().split(' ')
    f, p, a = None, None, None

    # Iterate over sentence
    for i in range(len(sentence)):
        # Discard very short words
        word = str(sentence[i])
        if len(word) < 4:
            continue

        # Check for special food types:
        if Levenshtein.distance(word, "north") < 3 and Levenshtein.distance(str(sentence[min(len(sentence) - 1, i+1)]), "american") < 3:
            f = "north american"
        elif Levenshtein.distance(word, "modern") < 3 and Levenshtein.distance(str(sentence[min(len(sentence) - 1, i+1)]), "european") < 3:
            f = "modern european"
            
        # Look for exact words or structures
        elif word in food_types and not f:
            f = word
        elif (Levenshtein.distance(word, "food") < 3 or Levenshtein.distance(word, "restaurant") < 3) and not f:
            food = str(sentence[i-1])
            if len(food) < 4:
                continue
            else:
                distances = []
                for j in food_types:
                    distances.append(Levenshtein.distance(food, j))
                if  len(food) == 4 and  min(distances) < 2:
                    f = food_types[np.argmin(distances)]
                elif min(distances) < 3:
                    f = food_types[np.argmin(distances)]
                    
        elif word in areas:
            # Check that is not north american type of food
            a = word
        elif (Levenshtein.distance(word, "area") < 3 or Levenshtein.distance(word, "part") < 3) and not a:
            area = str(sentence[i-1])
            if len(area) < 4:
                continue
            else:
                distances = []
                for j in areas:
                    distances.append(Levenshtein.distance(area, j))
                if len(area) == 4 and min(distances) < 2:
                    a = areas[np.argmin(distances)]
                elif min(distances) < 3:
                    a = areas[np.argmin(distances)]

        elif word in pricerange:
            p = word
        elif (Levenshtein.distance(word, "price") < 3 or  Levenshtein.distance(word, "restaurant") < 3) and not p:
            price = str(sentence[i-1])
            if len(price) < 4:
                continue
            else:
                distances = []
                for j in pricerange:
                    distances.append(Levenshtein.distance(price, j))
                if len(price) == 4 and min(distances) < 2:
                    p = pricerange[np.argmin(distances)]
                elif min(distances) < 3:
                    p = pricerange[np.argmin(distances)]

        # Since we have few possible areas and price ranges, check that the word is none of them
        else:
            if not p and word != "what":
                distances = []
                for j in pricerange:
                    distances.append(Levenshtein.distance(word, j))
                if min(distances) < 3 :
                    p = pricerange[np.argmin(distances)]
            if not a:
                distances = []
                for j in areas:
                    distances.append(Levenshtein.distance(word, j))
                if min(distances) < 3:
                    a = areas[np.argmin(distances)]
            
    return {"food": f, "price": p, "area":a}

def find_properties(sentence):
    sentence = sentence.lower().split(' ')
    consequent = {'touristic':0, 'romantic':0, 'children':0, 'seats':0}
    for i in range(len(sentence)):
        word = str(sentence[i])
        # Discard short words without information
        if len(word) < 5:
            continue

        # Check for expressions with more than one word
        elif Levenshtein.distance(word, "assigned") < 3 and Levenshtein.distance(str(sentence[min(len(sentence) - 1, i + 1)]), "seats") < 3:
            consequent["seats"] = 1
        # Check for other words, the shortest the word the less Levenshtein distance we apply since there is less chances  of commiting mistakes
        elif Levenshtein.distance(word, "touristic") < 3:
            consequent["touristic"] = 1
        elif Levenshtein.distance(word, "romantic") < 3:
            consequent["romantic"] = 1
        elif Levenshtein.distance(word, "children") < 3 or Levenshtein.distance(word, "kids") < 2:
            consequent["children"] = 1
    return consequent
    
def import_data(filepath):
    labels = []
    sentences = []

    with open(filepath, 'r') as file:
        for line in file:
            label, sentence = line.split(' ', 1)
            labels.append(label)
            sentences.append(sentence.strip())
    return labels, sentences

labels, sentences = import_data("../data/dialog_acts.dat")
label_mapping = dict(enumerate(np.unique(labels)))
label_mapping = {y: x for x, y in label_mapping.items()}
label_inverse_mapping = dict(zip(list(label_mapping.values()), list(label_mapping.keys())))

def labels_decode(labels):
    return np.array([label_inverse_mapping[label] for label in labels])

class bag_of_words_encoder:
    # Init the class with a corpus so we can initialize the encoder
    # By ussing CountVectorizer we ensure that the words not seen on the corpus when doing the model fit will be represented as a 0
    def __init__(self, corpus):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit_transform(corpus)

    # Method to encode a given text:
    def encode(self,data):
        return self.vectorizer.transform(data)

with open('../models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

classification_model = keras.models.load_model("../models/DL_model.keras")

def classify (utterance):
    sentence = vectorizer.encode([utterance.lower()])
    try:
        classification = labels_decode(np.argmax(classification_model.predict(sentence, verbose = 0), axis=1))
    except:
        classification = "unknown"
    return classification


def restaurant_rec(preferences):

    restaurants = pd.read_csv("../data/restaurant_info.csv")

    # Apply filters dynamically based on the non-empty values in preferences
    mask = pd.Series([True] * len(restaurants))

    if preferences.get('food'):
        mask &= (restaurants["food"] == preferences['food'])
    if preferences.get('price'):
        mask &= (restaurants["pricerange"] == preferences['price'])
    if preferences.get('area'):
        mask &= (restaurants["area"] == preferences['area'])

    # Find possible restaurants based on the constructed mask
    possible_restaurants = restaurants[mask]

    if(len(possible_restaurants) != 0):
        # Return the first matching restaurant
        return possible_restaurants.iloc[0]
    else:
        # Return an empty pandas.Series with the same index as the DataFrame columns
        return pd.Series([False] * len(restaurants.columns), index=restaurants.columns)


