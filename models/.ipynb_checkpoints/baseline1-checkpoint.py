# Count the different labels and return the count & total lines of the data file.
def count_labels(filepath):
    label_count = {}
    total_lines = 0

    with open(filepath, 'r') as file:
        for line in file:
            words = line.split(" ")
            total_lines += 1

            if words:
                label = words[0]
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1

        return total_lines, label_count

# Go over all lines, guess the majority class, and keep track of stats for the evaluation
def baseline1(test_filepath, guessed_label):
    total_lines = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    with open(test_filepath, 'r') as file:
        for line in file:
            total_lines += 1
            words = line.split(" ")
            correct_label = words[0].strip()

            if correct_label == guessed_label:
                true_positive += 1
            else:
                false_negative += 1

            if correct_label != guessed_label:
                false_positive += 1
                true_negative += 1

    # Compute accuracy, precision, recall, and F1 score for comparison 
    accuracy = (true_positive + true_negative) / total_lines if total_lines else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

    return accuracy, precision, recall, f1_score

# Locate the splitted data files
train_filepath = "data/train_part.dat"
test_filepath = "data/test_part.dat"

# Perform a label count and sort based on the counts (highest first)
total_lines, label_count = count_labels(train_filepath)
sorted_labels = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
most_occurred_label, highest_count = sorted_labels[0]

# Print the majority class for debugging
print(f"Majority class: {most_occurred_label} ({round((highest_count/total_lines)*100, 2)}%)")

# Execute the baseline1 model
accuracy, precision, recall, f1_score = baseline1(test_filepath, most_occurred_label)

