import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

"""
The following file performs an analysis on the utterance dataset
"""
# Function to check missing values
def missing_values(data):
    # Count missing values based on columns
    mis_val = pd.isnull(data).sum()
    # Calculate the proportion of missing values
    print(f'Missing values: {mis_val}, ( {np.round(100 * mis_val / len(data), 2)} %)')

# Function to check literal duplicates
def duplicate_values(data):
    texts, counts = np.unique(data, return_counts=True)
    duplicates = np.sum(counts > 1)
    print(f'Duplicate data: {duplicates}, ( {np.round(100 * duplicates / len(data), 2)} %)')

def perform_data_analysis():
    # Load data
    train_filepath = "../data/train_part.dat"
    train_dedup_filepath = "../data/deduplicated_train_part.dat"
    test_filepath = "../data/test_part.dat"

    y_train, x_train = import_data(train_filepath)
    y_dedup_train, x_dedup_train = import_data(train_dedup_filepath)
    y_test, x_test = import_data(test_filepath)

    print("Train set: ", len(x_train), len(y_train))
    print("Train set without duplicates: ", len(x_dedup_train), len(y_dedup_train))
    print("Test set: ", len(x_test), len(y_test))

    # Visualize labels distributions
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    train_dedup_labels, train_dedup_counts = np.unique(y_dedup_train, return_counts=True)
    test_labels, test_counts = np.unique(y_test, return_counts=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].bar(train_labels, train_counts)
    ax[0].set_title("Labels in train set")
    ax[0].set_xticklabels(train_labels, rotation=90)
    ax[1].bar(train_dedup_labels, train_dedup_counts)
    ax[1].set_title("Labels in train set without duplicates")
    ax[1].set_xticklabels(train_dedup_labels, rotation=90)
    ax[2].bar(test_labels, test_counts)
    ax[2].set_title("Labels in test set")
    ax[2].set_xticklabels(test_labels, rotation=90)
    plt.show()

    # Visualize utterances' length distributions
    train_lengths = (lambda x: [len(i.split(' ')) for i in x])(x_train)
    test_lengths = (lambda x: [len(i.split(' ')) for i in x])(x_test)

    fig, ax = plt.subplots(2, 2, figsize=(13, 12))
    ax[0, 0].hist(train_lengths, bins=20)
    ax[0, 0].set_ylabel('Frequency')
    ax[0, 0].set_xlabel('Words/utterance')
    ax[0, 0].set_title('Train set histogram')
    ax[0, 1].boxplot(train_lengths)
    ax[0, 1].set_ylabel('Words/utterance')
    ax[0, 1].set_title('Train set box plot')
    ax[1, 0].hist(test_lengths, bins=20)
    ax[1, 0].set_ylabel('Frequency')
    ax[1, 0].set_xlabel('Words/utterance')
    ax[1, 0].set_title('Test set histogram')
    ax[1, 1].boxplot(test_lengths)
    ax[1, 1].set_ylabel('Words/utterance')
    ax[1, 1].set_title('Test set box plot')
    plt.show()

    # Analyse out of vocabulary words
    train_words, train_counts = np.unique((lambda x: [word for sentence in x for word in sentence.split(' ')])(x_train),
                                          return_counts=True)
    test_words, test_counts = np.unique((lambda x: [word for sentence in x for word in sentence.split(' ')])(x_test),
                                        return_counts=True)
    out_vocabulary, out_vocabulary_counts = [], []

    for word, count in zip(test_words, test_counts):
        if word not in train_words:
            out_vocabulary.append(word)
            out_vocabulary_counts.append(count)

    print('Ammount of unique words in train set: ', len(train_words))
    print('Ammount of unique words in test set: ', len(test_words))
    print('Ammount of out of vocabulary words in test set: ', len(out_vocabulary))

    plt.boxplot(out_vocabulary_counts)
    plt.title("Out of vocabulary words frequencies in test set")
    plt.show()

    # Check missing valeus and literal duplicates
    print('-- Train set --')
    print('Utterances:')
    missing_values(x_train)
    print('Labels:')
    missing_values(y_train)
    print('Utterances:')
    duplicate_values(x_train)

    print('-- Test set --')
    print('Utterances:')
    missing_values(x_test)
    print('Labels:')
    missing_values(y_test)
    print('Utterances:')
    duplicate_values(x_test)

# Perform data analysis
perform_data_analysis()


