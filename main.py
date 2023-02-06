import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup

from model import BertClassifier
from dataset import NewsDataset
from train import train
from test import calculate_probs, evaluate


MODEL_PATH = './models/fake_news_classifier.pth'
MAX_LENGTH = 400
BATCH_SIZE = 16
EPOCHS = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOSS_FN = nn.CrossEntropyLoss()



def initialize_model(train_dataloader):
    model = BertClassifier(config=AutoConfig.from_pretrained('distilbert-base-uncased'), finetune = True)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    return model, optimizer, scheduler


def save_model(model, optimizer, scheduler):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, MODEL_PATH)


def load_model(model):
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def model_training():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    news = pd.read_csv('input/kaggle4/WELFake_Dataset.csv')
    news = news[~news.text.isnull()]

    news = news[['text', 'label']]
    news = news.astype({"label": int})

    X = news.text.values
    y = news.label.values

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=1234)


    train_dataset = NewsDataset(
        contents = X_train,
        labels = y_train,
        tokenizer = tokenizer,
        max_length = MAX_LENGTH)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = BATCH_SIZE)


    test_dataset = NewsDataset(
        contents = X_test,
        labels = y_test,
        tokenizer = tokenizer,
        max_length = MAX_LENGTH)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = BATCH_SIZE)

    ### TRAINING ###
    model, optimizer, scheduler = initialize_model(train_dataloader)
    train(model, optimizer, scheduler, train_dataloader, DEVICE, LOSS_FN, epochs = EPOCHS)

    ### SAVE ###
    save_model(model, optimizer, scheduler)


def test_kaggle1(model, tokenizer):
    fake_news = pd.read_csv('input/kaggle1/Fake.csv')
    true_news = pd.read_csv('input/kaggle1/True.csv')

    fake_news['label'] = np.ones(len(fake_news))
    true_news['label'] = np.zeros(len(true_news))

    test_news = pd.concat([fake_news, true_news])
    test_news = test_news[['text', 'label']]

    test_X = test_news.text.values
    test_y = test_news.label.values

    test_dataset = NewsDataset(
        contents = test_X,
        labels = test_y,
        tokenizer = tokenizer,
        max_length = MAX_LENGTH)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = BATCH_SIZE)

    fake_probs = calculate_probs(model, test_dataloader, DEVICE)[:, 1]
    y_pred = np.where(fake_probs >= 0.5, 1, 0)
    evaluate(test_y, y_pred)


def test_kaggle2(model, tokenizer):
    test_news = pd.read_csv('input/kaggle2/Test.csv')
    test_news.loc[test_news.label == "FAKE", 'label'] = 1
    test_news.loc[test_news.label == "REAL", 'label'] = 0
    test_news = test_news.astype({"label": int})
    test_news = test_news[['text', 'label']]

    test_X = test_news.text.values
    test_y = test_news.label.values

    test_dataset = NewsDataset(
        contents = test_X,
        labels = test_y,
        tokenizer = tokenizer,
        max_length = MAX_LENGTH)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = BATCH_SIZE)

    fake_probs = calculate_probs(model, test_dataloader, DEVICE)[:, 1]
    y_pred = np.where(fake_probs >= 0.5, 1, 0)
    evaluate(test_y, y_pred)


def test_kaggle3(model, tokenizer):
    test_news = pd.read_csv('input/kaggle3/train.csv')
    test_news = test_news.astype({"label": int})
    test_news = test_news[['text', 'label']]
    test_news = test_news[~test_news.text.isnull()]
    
    test_X = test_news.text.values
    test_y = test_news.label.values

    test_dataset = NewsDataset(
        contents = test_X,
        labels = test_y,
        tokenizer = tokenizer,
        max_length = MAX_LENGTH)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = BATCH_SIZE)

    fake_probs = calculate_probs(model, test_dataloader, DEVICE)[:, 1]
    y_pred = np.where(fake_probs >= 0.5, 1, 0)
    evaluate(test_y, y_pred)


def model_testing():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = BertClassifier(config=AutoConfig.from_pretrained('distilbert-base-uncased'), finetune = False)
    model.to(DEVICE)
    load_model(model)

    print("TESTING KAGGLE 1...")
    test_kaggle1(model, tokenizer)

    print("TESTING KAGGLE 2...")
    test_kaggle2(model, tokenizer)

    print("TESTING KAGGLE 3...")
    test_kaggle3(model, tokenizer)



if __name__ == "__main__":
    #model_training()

    model_testing()