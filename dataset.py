import torch
from torch.utils.data import Dataset

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import re


class NewsDataset(Dataset):
    def __init__(self, contents, labels, tokenizer, max_length):
        super(NewsDataset, self).__init__()
        self.contents = contents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text = self._clean_text(self.contents[index]),
            add_special_tokens = True,
            max_length = self.max_length,
            pad_to_max_length = True,
            return_attention_mask = True
        )

        ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(self.labels[index], dtype = torch.long)
            }


    def _clean_text(self, content):
        content = content.lower()
        
        content = re.sub(r"\'t", " not", content)
        content = re.sub(r'(@.*?)[\s]', ' ', content)

        content = re.sub(r'([\;\:\|•«\n])', ' ', content)
        content = " ".join([word for word in content.split()
                    if word not in stopwords.words('english')
                    or word in ['not', 'can']])

        content = re.sub(r'\s+', ' ', content).strip()

        return content