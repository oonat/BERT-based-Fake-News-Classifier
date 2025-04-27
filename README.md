
# Introduction
Fake News Classifier is a BERT-based model implemented as a part of my graduation project named "Skepsis". The main hypothesis behind this project was that the bodies of news articles have textual characteristics (e.g., formality, sentimental information, etc.) which might be useful to identify whether an article is unreliable or not.

# How to Use

## Dataset
The data used in this repo were taken from the Kaggle. As I am not the owner of the data, I cannot store it in the repo. However, one can download the data by following the steps given below:
1. Create a directory named **input** under the root directory.
2. Create folders named **kaggle1**, **kaggle2**, **kaggle3**, and **kaggle4** under the **input** directory.
3. Download the csv files from the Kaggle links shared below and save them under the corresponding directories:
- **kaggle1** -> https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **kaggle2** -> https://www.kaggle.com/datasets/amirmotefaker/detecting-fake-news-dataset (_rename news.csv to Test.csv_)
- **kaggle3** -> https://www.kaggle.com/datasets/marwaelsayedkhalil/fake-news-dataset
- **kaggle4** -> https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

The final folder structure should look like this:

- inputs/
  - kaggle1/
    - Fake.csv
    - True.csv
  - kaggle2/
    - Test.csv
  - kaggle3/
    - test.csv
    - train.csv
  - kaggle4/
    - WELFake_Dataset.csv


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
