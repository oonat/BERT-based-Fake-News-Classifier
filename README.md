
# Introduction
Fake News Classifier is a BERT-based model implemented as a part of my graduation project named "Skepsis". The main hypothesis behind this project was that the bodies of news articles have textual characteristics (e.g., formality, sentimental information, etc.) which might be useful to identify whether an article is unreliable or not.

# How to Use

## Installation

Before installing the required libraries, It is recommended to create a virtual environment.

The libraries required for the project are listed in the **requirements.txt** file. To download and install the necessary libraries,
```sh
pip install -r requirements.txt
```

## Model Training
We finetuned the "distilbert-base-uncased" model which can be found at https://huggingface.co/distilbert-base-uncased using four different Kaggle datasets created for the fake news classification task. Due to the file size limitations of Github, I was not able to add the training data used. Please contact me if you want to use this data.

The necessary functions for model training and testing can be found in the **main.py** file.

After the model training, one can save and load the trained model using the **save_model** and **load_model** functions given in **main.py**.

## Classification

The example code snippets for classification can be found inside **main.py**. Especially for multiple data, It would be useful to use the NewsDataset class implemented inside the **dataset.py**.

For server-side or single-query operations, please check the FakeNewsClassifier class given inside the **classify.py**. This file includes the production-level code that we used in the final form of our project.
