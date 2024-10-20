from utils import *
import keras
import matplotlib.pyplot as plt

"""
The following file contains the error analysis on the final classification system on the test set.
"""

# Function than given a model, performs an error analysis on the test set
def error_evaluation(path_to_model):
    # Load and encode test data
    test_filepath = "../data/test_part.dat"
    train_filepath = "../data/train_part.dat"

    y_train, x_train = import_data(train_filepath)
    y_test, x_test = import_data(test_filepath)
    vectorizer = bag_of_words_encoder(x_train)
    y_test_encode, x_test_encode = labels_encode(y_test), vectorizer.encode(x_test).toarray()

    # Load model and predict test
    model = keras.models.load_model(path_to_model)
    pred = np.argmax(model.predict(x_test_encode), axis=1)

    # Get wrongly classified utterances
    misclass_indexes = np.argwhere(pred != y_test_encode)
    misclass_utterances = np.array(x_test)[misclass_indexes]
    misclass_labels = np.array(y_test)[misclass_indexes]

    print("Ammount of missclassified utterances: ", len(misclass_indexes))
    print(f"Percentatge of missclassified utterances: {np.round(100 * (len(misclass_indexes) / len(y_test)), 2)}%")

    # See missclassified length distributions
    misclass_lengths = (lambda x: [len(str(i).split(' ')) for i in x])(misclass_utterances)
    plt.hist(misclass_lengths, bins=10)
    plt.ylabel('Frequency')
    plt.xlabel('Words/missclassied utterance')
    plt.title('Length distribution of missclassied utterances')
    plt.show()

    # See labels of missclassied utterances
    missclas_uq_labels, missclass_uq_labels_count = np.unique(misclass_labels, return_counts=True)

    plt.bar(missclas_uq_labels, missclass_uq_labels_count)
    plt.title("Misclassified utterances' labels")
    plt.xticks(missclas_uq_labels, rotation=90)
    plt.show()

    # Print list of missclassified utterances and their predicted labels
    print('MISCLASSIED UTTERANCES with predicted labels:')
    print(np.concatenate((misclass_utterances, labels_decode(pred)[misclass_indexes]), axis=1))

# Run error analysis on final model
error_evaluation("DL_model.h5")
