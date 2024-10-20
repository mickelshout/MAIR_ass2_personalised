# MAIR_2024

The following folder contains all the work done on the first project of Methods in Artificial Intelligence Research. 

There are 3 main directories:

- data: where all the files containing relevant data are stored
- dialog: has all files related to the dialogue system (Part 1b & 1c)
- models: has all files related to the utterance classification models (Part 1a)
- 
Enough to execute the dialogue system developed for Part 1b the user can simply open the terminal or command prompt on the current directory and execute the line: 
    
    python dialog/Interface.py 

To execute any of the files in "models" you can open the specific file on an IDE and run it or either execute:

    python models/name_of_the_file.py 
Each of the files executes one pipeline: the data exploration, the baseline system, the machine learning or the final model evaluation. More specific explanation can be found in each individual file.

To be able to execute it, the code has to be run in a python environment with the libraries:

- pandas 
- numpy 
- tensorflow
- keras
- scikit-learn
- python-Levenshtein
- pickle
