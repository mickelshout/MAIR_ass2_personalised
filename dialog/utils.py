import numpy as np
import pandas as pd
import Levenshtein
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from pathlib import Path


# Get the directory of the current script
script_dir = Path(__file__).parent

restaurant_info_path = script_dir.parent / 'data' / 'restaurant_info.csv'

df = pd.read_csv(restaurant_info_path)
food_types = list(np.unique(df.food)) + ['greek']
pricerange = np.unique(df.pricerange)
areas = list(np.unique(df.area.fillna('unknown')))
areas.remove("unknown")

"""
The folowing function retrives information given sentence from the user. It looks for food kind, pricerande and location.
"""

def find_preferences(sentence, levenshtein_distance):
    sentence = sentence.lower().split(' ')
    f, p, a = None, None, None

    # Iterate over sentence
    for i in range(len(sentence)):
        # Discard very short words
        word = str(sentence[i])
        if len(word) < 4:
            continue

        # Check for special food types:
        if Levenshtein.distance(word, "north") < levenshtein_distance and Levenshtein.distance(str(sentence[min(len(sentence) - 1, i+1)]), "american") < levenshtein_distance:
            f = "north american"
        elif Levenshtein.distance(word, "modern") < levenshtein_distance and Levenshtein.distance(str(sentence[min(len(sentence) - 1, i+1)]), "european") < levenshtein_distance:
            f = "modern european"

        # Look for exact words or structures
        elif word in food_types and not f:
            f = word
        elif (Levenshtein.distance(word, "food") < levenshtein_distance or Levenshtein.distance(word, "restaurant") < levenshtein_distance) and not f:
            food = str(sentence[i-1])
            if len(food) < 4:
                continue
            else:
                distances = []
                for j in food_types:
                    distances.append(Levenshtein.distance(food, j))
                if  len(food) == 4 and  min(distances) < 2:
                    f = food_types[np.argmin(distances)]
                elif min(distances) < levenshtein_distance:
                    f = food_types[np.argmin(distances)]
                    
        elif word in areas:
            # Check that is not north american type of food
            a = word
        elif (Levenshtein.distance(word, "area") < levenshtein_distance or Levenshtein.distance(word, "part") < levenshtein_distance) and not a:
            area = str(sentence[i-1])
            if len(area) < 4:
                continue
            else:
                distances = []
                for j in areas:
                    distances.append(Levenshtein.distance(area, j))
                if len(area) == 4 and min(distances) < levenshtein_distance:
                    a = areas[np.argmin(distances)]
                elif min(distances) < levenshtein_distance:
                    a = areas[np.argmin(distances)]

        elif word in pricerange:
            p = word
        elif (Levenshtein.distance(word, "price") < levenshtein_distance or  Levenshtein.distance(word, "restaurant") < levenshtein_distance) and not p:
            price = str(sentence[i-1])
            if len(price) < 4:
                continue
            else:
                distances = []
                for j in pricerange:
                    distances.append(Levenshtein.distance(price, j))
                if len(price) == 4 and min(distances) < levenshtein_distance:
                    p = pricerange[np.argmin(distances)]
                elif min(distances) < levenshtein_distance:
                    p = pricerange[np.argmin(distances)]

        # Since we have few possible areas and price ranges, check that the word is none of them
        else:
            if not p and word != "what":
                distances = []
                for j in pricerange:
                    distances.append(Levenshtein.distance(word, j))
                if min(distances) < levenshtein_distance :
                    p = pricerange[np.argmin(distances)]
            if not a:
                distances = []
                for j in areas:
                    distances.append(Levenshtein.distance(word, j))
                if min(distances) < levenshtein_distance:
                    a = areas[np.argmin(distances)]
            
    return {"food": f, "price": p, "area":a}

def find_properties(sentence, levenshtein_distance):
    sentence = sentence.lower().split(' ')
    consequents = []
    for i in range(len(sentence)):
        word = str(sentence[i])
        # Discard short words without information
        if len(word) < 5:
            continue

        # Check for expressions with more than one word
        elif Levenshtein.distance(word, "assigned") < levenshtein_distance and Levenshtein.distance(str(sentence[min(len(sentence) - 1, i + 1)]), "seats") < levenshtein_distance:
            consequents.append("seats")
        # Check for other words, the shortest the word the less Levenshtein distance we apply since there is less chances  of commiting mistakes
        elif Levenshtein.distance(word, "touristic") < levenshtein_distance:
            consequents.append("touristic")
        elif Levenshtein.distance(word, "romantic") < levenshtein_distance:
            consequents.append("romantic")
        elif Levenshtein.distance(word, "children") < levenshtein_distance or Levenshtein.distance(word, "kids") < levenshtein_distance:
            consequents.append("children")
    return consequents

# Function that imports and extracts the labels and sentences from a given path
def import_data(filepath):
    labels = []
    sentences = []

    with open(filepath, 'r') as file:
        for line in file:
            label, sentence = line.split(' ', 1)
            labels.append(label)
            sentences.append(sentence.strip())
    return labels, sentences

dialogs_path = script_dir.parent / 'data' / 'dialog_acts.dat'

# Labels dictionary
labels, sentences = import_data(dialogs_path)
label_mapping = dict(enumerate(np.unique(labels)))
label_mapping = {y: x for x, y in label_mapping.items()}
label_inverse_mapping = dict(zip(list(label_mapping.values()), list(label_mapping.keys())))

# Function that given a list of numerical labels decodes them into categorical
def labels_decode(labels):
    return np.array([label_inverse_mapping[label] for label in labels])

# Class that given a corpus of sentences, fits a bag of words model so it can be used to encode new texts
class bag_of_words_encoder:
    # Init the class with a corpus so we can initialize the encoder
    # By ussing CountVectorizer we ensure that the words not seen on the corpus when doing the model fit will be represented as a 0
    def __init__(self, corpus):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit_transform(corpus)

    # Method to encode a given text:
    def encode(self,data):
        return self.vectorizer.transform(data)

model_path = script_dir.parent / 'models' / 'DL_model.h5'

# Get the pretrained model
classification_model = keras.models.load_model(model_path)

vectorizer_path = script_dir.parent / 'models' / 'vectorizer.pkl'

# Get vectorizer corresponging to the model
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Function that classifies the utterance from the user using the classification model
def classify (utterance):
    sentence = vectorizer.encode([utterance.lower()])
    try:
        classification = labels_decode(np.argmax(classification_model.predict(sentence, verbose = 0), axis=1))
    except:
        classification = "unknown"
    return classification


def get_recommendations(preferences, additional_preferences):
    additional_preferences = {key: True for key in additional_preferences}
    data = pd.read_csv(restaurant_info_path)
    mask = pd.Series([True] * len(data))

    # Check if all preferences are either None or "don't mind"
    if all(value is None or value.lower() in ["don't mind", "dont mind", "don't care", "dont care"] for value in
           preferences.values()):
        # If all preferences are "don't mind", return a random restaurant
        random_recommendation = data.sample(n=1)
        reasoning = {
            random_recommendation.index[0]: ["Random recommendation because you didn't provide specific preferences."]}
        return random_recommendation, reasoning

    # Store the reasoning for all restaurants
    reasoning = {index: [] for index in data.index}

    # Basic Preferences
    if preferences.get("food"):
        mask &= (data["food"] == preferences["food"])
        for index in data.index:
            if data.at[index, "food"] == preferences["food"]:
                reasoning[index].append(f"It serves {preferences['food']} food, just like you wanted.")

    if preferences.get("price"):
        mask &= (data["pricerange"] == preferences["price"])
        for index in data.index:
            if data.at[index, "pricerange"] == preferences["price"]:
                reasoning[index].append(f"The price range is {preferences['price']}, matching your budget.")

    if preferences.get("area"):
        mask &= (data["area"] == preferences["area"])
        for index in data.index:
            if data.at[index, "area"] == preferences["area"]:
                if preferences["area"] == "center":
                    reasoning[index].append(f"It's located right in the center of town.")
                else:
                    reasoning[index].append(f"\nIt's in the {preferences['area']} part of town, which you prefer.")

    # Additional Preferences
    if "romantic" in additional_preferences and additional_preferences["romantic"]:
        mask &= (~data["crowdedness"]) & (data["length_stay"])
        for index in data.index:
            if not data.at[index, "crowdedness"] and data.at[index, "length_stay"]:
                reasoning[index].append(
                    "It's perfect for a romantic evening—quiet and ideal for a long, relaxing stay.")

    if "touristic" in additional_preferences and additional_preferences["touristic"]:
        mask &= (data["quality"]) & (data["pricerange"] == 'cheap') & (data["food"] != 'Romanian')
        for index in data.index:
            if data.at[index, "quality"] and data.at[index, "pricerange"] == 'cheap' and data.at[
                index, "food"] != 'Romanian':
                reasoning[index].append("It's a great spot for tourists, offering good and affordable food.")

    if "children" in additional_preferences and additional_preferences["children"]:
        mask &= (~data["length_stay"])
        for index in data.index:
            if not data.at[index, "length_stay"]:
                reasoning[index].append("It's family-friendly, with a shorter stay ideal for children.")

    if "seats" in additional_preferences and additional_preferences["seats"]:
        mask &= (data["crowdedness"])
        for index in data.index:
            if data.at[index, "crowdedness"]:
                reasoning[index].append(
                    "It has assigned seating, so you won’t have to worry about finding a place when it's busy.")

    # Get recommended restaurants that match the preferences
    recommended_restaurants = data[mask]

    # Only keep the restaurants that match the preferences
    recommended_reasoning = {index: reasoning[index] for index in recommended_restaurants.index if index in reasoning}

    # Combine the reasoning into a single paragraph for each restaurant
    combined_reasoning = {index: " ".join(reasoning[index]) for index in recommended_reasoning}

    return recommended_restaurants, combined_reasoning


#Helper function for the all caps feature
def format_output(text, caps):
    return text.upper() if caps else text
