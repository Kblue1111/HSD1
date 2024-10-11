# SCOPE

> SCOPE: Hybrid Optimization Strategy for Higher-order Mutation-based Fault Localization

## Introduction

This project corresponds to the experiments  in the paper `SCOPE: Hybrid Optimization Strategy for Higher-order Mutation-based Fault Localization`.

## Dataset

We conduct our experiments on Defects4J (2.0.0), which
contains reproducible real-world faults from a wide range of
software projects and has been widely used for previous FL
research. 

The data for this experiment can be downloaded from https://github.com/rjust/defects4j/releases/tag/v2.0.0.

## How To Use

`mutationTool`directory contains operations that modify the source program to generate higher-order mutants. You can perform the mutation operation by running `main.py`.

- `execute` directory includes mutation operations for both first-order mutants and higher-order mutants. This includes generating mutants and executing test cases to obtain kill information.

- `logs` directory contains log information from the program's execution. 
- `tool` directory includes the calculation process for evaluation metrics such as EXAM, MAR, and MFR.

You can configure the local paths of different files in `config.json`.



`trainModelSOM` directory contains files for training models and predicting mutation outcomes. You can execute these by running `main.py`.

- In `main.py`, the function `logger_config(log_path)` configures the output format and location for logging.
- The functions `dstar(Akf, Anf, Akp, Anp)` and `ochiai(Akf, Anf, Akp, Anp)` are the calculation formulas for MBFL.
- The function `getHugeLine(project, version, info)` retrieves the line number of a mutant.
- The function `getStaticFeature(info, line_num, staticFea)` obtains static features based on the mutant’s information and standard line number.
- The function `getFeatureVect(project, version, info, result, staticFea, dynamicFea, crFea, sfFea, ssFea)` generates the feature vector for a specific version of a mutant.
- The function `getTrainData(project, version)` gathers data for the training and testing sets.
- The function `data_split(featureVectList, labelList, train_ratio=0.1)` divides the training and testing data based on different sampling rates.
- The function `randomForestModel(featureVectList, labelList, saveFile, project, version, mode, kill_data, i)` loads a random forest model, and other models are loaded in a similar fashion.
- The function `executeModel` is used for model execution and prediction.
