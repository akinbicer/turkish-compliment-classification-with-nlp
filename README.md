# Turkish Compliment Classification with NLP

This repository contains a Python script for training, evaluating, and deploying a Natural Language Processing (NLP) model for classifying compliments in Turkish language. The model is trained using the Multinomial Naive Bayes algorithm and is saved in ONNX and Joblib formats for deployment.

## Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Model Training](#model-training)

## Introduction

The goal of this project is to build a machine learning model that can classify compliments as either "Male" (0) or "Female" (1) based on their content. The model uses TF-IDF vectorization for text data and is trained on a dataset of compliments in Turkish language.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.6+
- pandas
- joblib
- numpy
- skl2onnx
- scikit-learn

You can install these dependencies using pip:

```
pip install pandas joblib numpy skl2onnx scikit-learn
```

## Usage

1. Clone this repository to your local machine:

```
git clone https://github.com/akinbicer/turkish-compliment-classification-with-nlp.git
```

2. Navigate to the project directory:

```
cd turkish-compliment-classification-with-nlp
```

3. Run the Python script to train and save the model:

```
python Complimenter.NLP.Classification.py
```

4. After training, the model will be saved in both ONNX and Joblib formats in the `Models` directory.

## Model Training

- The script `Complimenter.NLP.Classification.py` handles data loading, preprocessing, model selection, hyperparameter tuning, training, evaluation, and saving the model.
- The dataset is loaded from a CSV file (`Turkish.csv`) containing compliments and their corresponding labels (0 for Male, 1 for Female).
- TF-IDF vectorization is used to convert text data into numerical features.
- The Multinomial Naive Bayes algorithm is chosen as the classification model.
- Hyperparameter tuning is performed using Grid Search with cross-validation.
- Model evaluation metrics such as accuracy and classification report are displayed.

## License
This project is distributed under the MIT license. Refer to the [LICENSE](LICENSE) file for more information.

## Issues, Feature Requests or Support
Please use the [New Issue](https://github.com/akinbicer/turkish-compliment-classification-with-nlp/issues/new) button to submit issues, feature requests or support issues directly to me. You can also send an e-mail to akin.bicer@outlook.com.tr.
