import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers import DistilBertModel


class BertClassifier(PreTrainedModel):
    def __init__(self, config, finetune = True):
        super(BertClassifier, self).__init__(config)

        D_in, H, D_out = 768, 50, 2

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if not finetune:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits

