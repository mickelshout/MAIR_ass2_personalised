import math

def split_dat_file(filepath, train_filepath, test_filepath, test_percentage=0.15):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        lines = [line.lower() for line in lines]

        total_lines = len(lines)
        test_size = math.ceil(total_lines * test_percentage)
        test_lines = lines[:test_size]
        train_lines = lines[test_size:]

        with open(test_filepath, 'w') as test_file:
            test_file.writelines(test_lines)

        with open(train_filepath, 'w') as train_file:
            train_file.writelines(train_lines)

        return total_lines, len(train_lines), len(test_lines)
    
filepath = "./data/dialog_acts.dat"
train_filepath = "./data/train_part.dat"
test_filepath = "./data/test_part.dat"

total_count, train_count, test_count = split_dat_file(filepath, train_filepath, test_filepath)

print(f"Total lines: {total_count}, train lines: {train_count}, test lines: {test_count}")