import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def calculate_probs(model, test_dataloader, device):
    model.eval()

    output_list = []

    for i, batch in enumerate(test_dataloader):

        print(f"Batch: {i}")

        batch_ids, batch_attention_mask =  batch['ids'], batch['attention_mask']

        with torch.no_grad():
            outputs = model(batch_ids.to(device), batch_attention_mask.to(device))
        output_list.append(outputs)
    

    output_list = torch.cat(output_list, dim=0)
    probs = F.softmax(output_list, dim=1).cpu().detach().numpy()

    return probs



def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    confusion_mat = confusion_matrix(y_true, y_pred)

    print(f'Accuracy: {accuracy * 100 :.2f}% | F1-Macro: {f1_macro :.3f}')
    print(confusion_mat)
