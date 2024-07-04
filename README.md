# HSD

The `mutationTool` directory contains operations for mutating the source program to generate higher-order mutants. You can perform the mutation operations by running `main.py`.

The `trainModelSOM` directory contains files for training models and predicting the execution results of mutants. You can run the program by executing `main.py`.

First, generate a set of mutants through the mutation operations, then divide them into a training set and a test set. Use the training set to train the model to predict the execution results of the mutants in the test set.