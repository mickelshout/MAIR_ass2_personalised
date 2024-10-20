from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import keras

"""
The following file defines a class that contains all the implemented machine learning classification class
It also runs all the ML pipeline, including model training, evaluation and user interaction
"""

class ML_classifiers:
    def __init__(self):
        print('-- Machine Learning Pipeline --')
        self.process_data()
        self.init_models()
        self.evualate_models()
        self.run_user_interaction()

    # Function to load and process all required data
    def process_data(self):
        # Load data, with and without duplicates
        print('Loading data')
        train_filepath = "../data/train_part.dat"
        dedup_train_filepath = "../data/deduplicated_train_part.dat"
        test_filepath = "../data/test_part.dat"
        self.y_train, self.x_train = import_data(train_filepath)
        self.y_dedup_train, self.x_dedup_train = import_data(dedup_train_filepath)
        self.y_test, self.x_test = import_data(test_filepath)

        # Process data
        print('Processing data')
        # Numerical labels
        self.y_train_encode, self.y_dedup_train_encode, self.y_test_encode = labels_encode(self.y_train), labels_encode(self.y_dedup_train), labels_encode(self.y_test)
        # One hot encode labels
        self.y_train_1hencode, self.y_dedup_train_1hencode, self.y_test_1hencode = one_hot_encode(self.y_train), one_hot_encode(self.y_dedup_train), one_hot_encode(self.y_test)
        # Encode text to be classified
        self.vectorizer = bag_of_words_encoder(self.x_train)
        self.x_train_encode, self.x_dedup_train_encode, self.x_test_encode = self.vectorizer.encode(self.x_train).toarray(), self.vectorizer.encode( self.x_dedup_train).toarray(), self.vectorizer.encode(self.x_test).toarray()

        # Since we have imbalanced data, we will assign a weight to each class so models don’t develop a bias towards the most common one.
        print('Calculating class weights')
        self.class_weights = compute_class_weights(self.y_train)
        self.dedup_weights = compute_class_weights(self.y_dedup_train)

    # Function to initialize and train all ML models
    def init_models(self):
        # Initialize and train ML models
        print('Model initialization and training')

        # Logistic Regression
        self.clf, self.clf_dedup = LogisticRegression(random_state=0,class_weight=self.class_weights,multi_class="multinomial"), LogisticRegression(random_state=0,class_weight=self.class_weights,multi_class="multinomial")
        self.clf.fit(self.x_train_encode, self.y_train_encode)
        self.clf_dedup.fit(self.x_dedup_train_encode, self.y_dedup_train_encode)

        # Decision Tree
        self.tree, self.tree_dedup = DecisionTreeClassifier(random_state=0, class_weight=self.class_weights), DecisionTreeClassifier(random_state=0, class_weight=self.class_weights)
        self.tree.fit(self.x_train_encode, self.y_train_encode)
        self.tree_dedup.fit(self.x_dedup_train_encode, self.y_dedup_train_encode)

        # Random Forest
        self.forest, self.forest_dedup = RandomForestClassifier(random_state=0, class_weight=self.class_weights), RandomForestClassifier(random_state=0, class_weight=self.class_weights)
        self.forest.fit(self.x_train_encode, self.y_train_encode)
        self.forest_dedup.fit(self.x_dedup_train_encode, self.y_dedup_train_encode)

        # Deep Learning
        self.dl_model = keras.Sequential()
        self.dl_model.add(keras.layers.Dense(256, input_shape=(718,)))
        self.dl_model.add(keras.layers.BatchNormalization())  # Add Batch Normalization
        self.dl_model.add(keras.layers.Activation("relu"))
        self.dl_model.add(keras.layers.Dense(100))
        self.dl_model.add(keras.layers.BatchNormalization())  # Add Batch Normalization
        self.dl_model.add(keras.layers.Activation("relu"))
        self.dl_model.add(keras.layers.Dense(15, activation='softmax'))
        self.dl_model.compile(loss='categorical_crossentropy',
                         optimizer="adam", metrics=['accuracy'])
        self.dl_model.fit(self.x_train_encode, self.y_train_1hencode, batch_size=500,
                     epochs=70, verbose=None, validation_split=0.2, class_weight=self.class_weights)

        # Saving the model for part 1b
        self.dl_model.save("../models/DL_model.h5")

        self.dedup_dl_model = keras.Sequential()
        self.dedup_dl_model.add(keras.layers.Dense(256, input_shape=(718,)))
        self.dedup_dl_model.add(keras.layers.BatchNormalization())  # Add Batch Normalization
        self.dedup_dl_model.add(keras.layers.Activation("relu"))
        self.dedup_dl_model.add(keras.layers.Dense(100))
        self.dedup_dl_model.add(keras.layers.BatchNormalization())  # Add Batch Normalization
        self.dedup_dl_model.add(keras.layers.Activation("relu"))
        self.dedup_dl_model.add(keras.layers.Dense(15, activation='softmax'))
        self.dedup_dl_model.compile(loss='categorical_crossentropy',
                               optimizer="adam", metrics=['accuracy'])
        self.dedup_dl_model.fit(self.x_dedup_train_encode, self.y_dedup_train_1hencode, batch_size=500,
                           epochs=70, verbose=None, validation_split=0.2, class_weight=self.dedup_weights)

    # Function to run an evaluation over all the trained models on the test set
    def evualate_models(self):
        # Logistic Regression
        print('-- Logisitic Regression evaluation --')
        pred = self.clf.predict(self.x_test_encode)
        evaluate_model(self.y_test_encode, pred)
        print('-- Logisitic Regression evaluation after deduplication--')
        pred = self.clf_dedup.predict(self.x_test_encode)
        evaluate_model(self.y_test_encode, pred)
        print()

        # Decision Tree
        print('-- Decision Tree evaluation --')
        pred = self.tree.predict(self.x_test_encode)
        metrics = evaluate_model(self.y_test_encode, pred)
        print('-- Decision Tree evaluation after deduplication--')
        pred = self.tree_dedup.predict(self.x_test_encode)
        metrics = evaluate_model(self.y_test_encode, pred)
        print()

        # Random Forest
        print('-- Random Forest evaluation --')
        pred = self.forest.predict(self.x_test_encode)
        metrics = evaluate_model(self.y_test_encode, pred)
        print('-- Random Forest evaluation after deduplication--')
        pred = self.forest_dedup.predict(self.x_test_encode)
        metrics = evaluate_model(self.y_test_encode, pred)
        print()

        # Deep Learning
        print('-- Deep Learning evaluation --')
        pred = self.dl_model.predict(self.x_test_encode)
        evaluate_model(np.argmax(self.y_test_1hencode, axis=1), np.argmax(pred, axis=1))
        print('-- Deep Learning evaluation after deduplication--')
        pred = self.dedup_dl_model.predict(self.x_test_encode)
        evaluate_model(np.argmax(self.y_test_1hencode, axis=1), np.argmax(pred,axis=1))

    # Function to run user interactive program to predict new utterances
    def run_user_interaction(self):

        print('-- User interaction on Machine Learning models --')
        # keep asking for user input and classify the entered sentence
        while True:
            print('Choose a ML model to predict a sentence: ')
            print(' . Logistic Regression: lr')
            print(' . Decision Trees: dt')
            print(' . Random Forest: rf')
            print(' . Feed forward neural network: ff')
            model = input('Write here the initials of the model or Q if you want to quit the program: ').lower()

            if model in ['lr', 'dt', 'rf', 'ff']:
                dedup = input('Write D if you want to use the version trained without dubles: ').lower()
                sentence = input("Enter a sentence: ")
                new_sentence = self.vectorizer.encode([sentence.lower()])
                if model == 'lr':
                    if dedup == 'd':
                        pred = self.clf_dedup.predict(new_sentence)
                    else:
                        pred = self.clf.predict(new_sentence)
                elif model == 'dt':
                    if dedup == 'd':
                        pred = self.tree_dedup.predict(new_sentence)
                    else:
                        pred = self.tree.predict(new_sentence)
                elif model == 'rf':
                    if dedup == 'rf':
                        pred = self.forest_dedup.predict(new_sentence)
                    else:
                        pred = self.forest.predict(new_sentence)
                else:
                    if dedup == 'd':
                        pred = np.argmax(self.dedup_dl_model.predict(new_sentence), axis=1)
                    else:
                        pred = np.argmax(self.dl_model.predict(new_sentence), axis=1)

                print(f'Prediction for "{sentence}" is: {labels_decode(pred)}"')

            elif model in ['q']:
                print("Exiting...")
                break
            else:
                print("The input was incorrect.\n")


# This line runs the machine learing pipeline
test = ML_classifiers()











