# helped a lot: https://www.linkedin.com/pulse/text-classification-using-bag-words-approach-nltk-scikit-rajendran
# and: https://scikit-learn.org/stable/modules/tree.html

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

# import and extract the labels and sentences
def import_data(filepath):
    labels = []
    sentences = []
    
    with open(filepath, 'r') as file:
        for line in file:
            label, sentence = line.split(' ', 1)
            labels.append(label)
            sentences.append(sentence.strip())
    return labels, sentences

# takes as input a sentence and tries to classify it
def classify_sentence(sentence):
    sentence_transformed = vectorizer.transform([sentence])
    numeric_prediction = classifier.predict(sentence_transformed)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    predicted_label = reverse_label_mapping[numeric_prediction[0]]
    return predicted_label

# import and seperate the data
labels, sentences = import_data("data/dialog_acts.dat")

# transfor the labels to numeric values
label_mapping = {label: idx for idx, label in enumerate(list(set(labels)))}
numeric_labels = [label_mapping[label] for label in labels]

# easy countvectorizer function to help with the bag of words model
vectorizer = CountVectorizer()
sparse_matrix = vectorizer.fit_transform(sentences)

# train the decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(sparse_matrix, numeric_labels)

# keep asking for user input and classify the entered sentence
while True:
    new_sentence = input("Enter a sentence: ")
    if new_sentence.lower() in ['q']:
        print("Exiting...")
        break

    # try to classify the new sentence. Maybe add some inputfiltering
    predicted_label = classify_sentence(new_sentence)
    print(f"Prediction for '{new_sentence}' is: {predicted_label}")