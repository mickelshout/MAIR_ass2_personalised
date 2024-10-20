"""
The following python file contains common functions we will use during model development and testing on text classification.
"""
import keras
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

# Labels dictionary
labels, sentences = import_data("../data/dialog_acts.dat")
label_mapping = dict(enumerate(np.unique(labels))) 
label_mapping = {y: x for x, y in label_mapping.items()}
label_inverse_mapping = dict(zip(list(label_mapping.values()), list(label_mapping.keys())))

# Function that given a list of labels, one hot encodes it
def one_hot_encode(labels):
    # Convert them to numerical representation and then to 1 hot encode
    numeric_labels = [label_mapping[label] for label in labels]
    numeric_labels = keras.utils.to_categorical(numeric_labels, len(label_mapping))
    return numeric_labels

# Function that given a list of categorical labels encodes them into numerical
def labels_encode(labels):
    return np.array([label_mapping[label] for label in labels])

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

# Function than given a list of labels, calculates the weigth of each class
def compute_class_weights(labels):
    unique_classes, class_counts = np.unique(np.array(labels), return_counts=True)
    total_samples = len(labels)
    class_weights = {label_mapping[cls]: total_samples / (len(unique_classes) * count)
                     for cls, count in zip(unique_classes, class_counts)}
    return class_weights

# Function to evaluate a model, given the predictions and true labels it returns evaluation metrics
def evaluate_model(labels, pred):
    accuracy = accuracy_score(labels, pred)
    precision = precision_score(labels, pred, average="macro")
    recall = recall_score(labels, pred, average="macro")
    f1 = f1_score(labels, pred, average="macro")
    print(f'    - Accuracy:{accuracy}')
    print(f'    - Macro precision:{precision}')
    print(f'    - Macro recall:{recall}')
    print(f'    - Macro F1 score:{f1}')
    return [accuracy, precision, recall, f1]

def deduplication(filepath, deduplication_filepath):
    with open(filepath, 'r') as file:
        new = list(set(file))
    with open(deduplication_filepath, 'w') as file:
        for sentence in new:
            file.writelines(sentence)





