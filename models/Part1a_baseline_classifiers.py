from utils import *
import numpy as np
"""
The following file defines a class that contains both baseline utterance classification systems:
One of them assigns the majority class among the training data and the other one follows a rule-based system
It also runs a pipeline including evaluation and user interaction.
"""

class Baseline_classifiers:
    def __init__(self):
        print('-- Baseline classifiers Pipeline --')
        self.majority_class = self.init_majority_class()
        self.evaluate_classifiers()
        self.run_user_interaction()

    def init_majority_class(self):
        # Get all train set labels
        train_filepath = "../data/train_part.dat"
        y_train, x_train = import_data(train_filepath)
        values, counts = np.unique(y_train, return_counts=True)
        return values[np.argmax(counts)]

    # Function to get majority class given an utterance
    def predict_majority_class(self, utterance):
        return self.majority_class

    # Function to predict an utterance class given a rule based model
    def predict_rule_based(self, utterance):
        # The rules we filter on
        classification_rules = {
            "thankyou": ['thankANDyou', 'thanks'],
            "bye": [],
            "ack": [],
            "affirm": ['yes', 'yea', 'yeah', 'right', 'correct'],
            "confirm": [],
            "deny": ['wrong'],
            "hello": ['hello', 'hi', 'halo'],
            "inform": ['i', 'im', 'spanish', 'english', 'chinese', 'thai', 'portuguese', 'bistro', 'cantonese',
                       'eritrean', 'hungarian', 'kosher', 'halal', 'catalan', 'turkish', 'jamaican', 'malaysian',
                       'danish', 'gastropub', 'german', 'chiquito', 'venetian', 'canapes', 'polish', 'afghan',
                       'singaporean', 'singapore', 'brazilian', 'scandinavian', 'irish', 'lebanese', 'seafood',
                       'japanese', 'korean', 'european', 'restaurant', 'greek', 'african', 'anything', 'vietnamese',
                       'expensive', 'cheap', 'east', 'north', 'west', 'center', 'australian', 'moderately', 'south',
                       'american', 'food', 'british', 'indian', 'italian', 'french', 'asian', 'oriental', 'mexican',
                       'swedish', 'india', 'mediterranean', 'any', 'matter', 'care', 'polynesian', 'moderate'],
            "negate": ['no'],
            "null": [],
            "repeat": ['repeat'],
            "reqalts": ['how', 'whatANDabout', 'howANDabout'],
            "reqmore": ['more'],
            "request": ['whats', 'where', 'whatNOTabout', 'post', 'phone', 'address', 'area', 'price', 'telephone',
                        'postcode', 'location'],
            "restart": ['start', 'reset']
        }

        classified = False
        classification = None
        line = str(utterance).lower().split(" ")

        # First we look for static checks and rules
        if (line[-1] == 'yes') and classified == False:
            classification ="affirm"
            classified = True

        if (line[-1] == 'bye') and classified == False:
            if (len(line) > 3):
                if (line[-2] == 'good' and line[-3] == 'you'):
                    classification = "thankyou"
                    classified = True
            else:
                classification = "bye"
                classified = True

        for rule in classification_rules['inform']:
            if rule in line:
                if (len(line) > 2):
                    if (line[0] == 'how' and line[1] == 'about'):
                        classification = "reqalts"
                        classified = True
                classification = "inform"
                classified = True

        # If no static rule was found, we try dynamic rules through all classes
        for key in list(classification_rules.keys()):
            if classified == False:
                rules = classification_rules[key]
                # Check all rules in a class
                for rule in rules:
                    if 'NOT' in rule:
                        rule = rule.split("NOT")
                        if rule[0] in line and rule[1] not in line and classified == False:
                            classified = True
                            classification = key
                    elif 'AND' in rule:
                        rule = rule.split("AND")
                        if rule[0] in line and rule[1] in line and classified == False:
                            classified = True
                            classification = key
                    elif rule in line and classified == False:
                        classified = True
                        classification = key

        # If no rule was found on the utterance, we will consider is an inform sentence
        if classification == None:
            classification = "null"
        return  classification


    # Function to run an evaluation over both baselines on the test set
    def evaluate_classifiers(self):
        # Load test data
        test_filepath = "../data/test_part.dat"
        y_test, x_test = import_data(test_filepath)

        # Majority class
        print('-- Majority Class baseline evaluation --')
        pred = [self.majority_class] * len(x_test)
        evaluate_model(y_test, pred)

        # Rule-based
        print('-- Rule Based baseline evaluation --')
        pred = [self.predict_rule_based(sentence) for sentence in x_test]
        evaluate_model(y_test, pred)

    # Function to run user interactive program to predict new utterances
    def run_user_interaction(self):
        print('-- User interaction on Baseline Models --')
        # keep asking for user input and classify the entered sentence
        while True:
            print('Choose a base line  model to predict a sentence: ')
            print(' . Majority class: mc')
            print(' . Rule based: rb')
            model = input('Write here the initials of the model or Q if you want to quit the program: ').lower()

            if model in ['mc', 'rb']:
                sentence = input("Enter a sentence: ")
                new_sentence = sentence.lower()
                if model == 'mc':
                    pred = self.predict_majority_class(new_sentence)
                elif model == 'rb':
                    pred = self.predict_rule_based(new_sentence)
                print(f'Prediction for "{sentence}" is: "{pred}"')

            elif model in ['q']:
                print("Exiting...")
                break
            else:
                print("The input was incorrect.\n")



# This line runs the baseline pipeline
test = Baseline_classifiers()