import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

import torch
import torch.nn.functional as F
from transformers import AutoConfig, DistilBertTokenizer

from model import BertClassifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class FakeNewsClassifier:
	def __init__(self, model_path, max_length):
		self._load_model(model_path)
		self.max_length = max_length


	def _load_model(self, model_path):
		self.model = \
			BertClassifier(config=AutoConfig.from_pretrained('distilbert-base-uncased'), finetune = False)

		self.model.to(DEVICE)

		checkpoint = torch.load(model_path)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


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


	def classify(self, content):
		self.model.eval()

		inputs = self.tokenizer.encode_plus(
					text = self._clean_text(content),
					add_special_tokens = True,
					max_length = self.max_length,
					pad_to_max_length = True,
					return_attention_mask = True
				)

		ids = inputs["input_ids"]
		ids = torch.tensor(ids, dtype=torch.long).to(DEVICE)
		ids = ids.unsqueeze(0)

		attention_mask = inputs["attention_mask"]
		attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(DEVICE)
		attention_mask = attention_mask.unsqueeze(0)

		output = self.model(ids, attention_mask)
		probs = F.softmax(output, dim=1).cpu().detach().numpy()

		model_result = "Fake" if probs[0][1] >= 0.5 else "True"

		return model_result




if __name__ == "__main__":

	model_path = 'fake_news_classifier.pth'
	max_length = 400
	classifier = FakeNewsClassifier(model_path, max_length)


	url = 'https://www.reuters.com/world/europe/eu-oil-embargo-in-days-ukraine-isolation-drives-russia-closer-china-2022-05-23/'
	content = """FORT MEADE, MD—Praised as a leader in centering people from groups not traditionally represented onscreen, the National Security Agency was honored Monday for its diversity in surveillance footage. “In a time when people from marginalized communities long ignored in the media have to fight for space on the screen, the NSA has used its vast network of surveillance cameras to center minority stories,” said Terrence Walz of the American Civil Liberties Union, presenting a plaque to NSA director Paul Nakasone and singling out the agency’s PRISM and Upstream initiatives for their focus on portraying people from minority groups in the environments in which they actually live. “Many organizations pay lip service to minority perspectives, but the NSA really understands the assignment. It routinely centers people from every background in its surveillance efforts—Black or white, gay or straight, people of every creed and ethnic group are prominently featured in the NSA’s work. In fact, there are many examples in which the NSA films only people from marginalized backgrounds, and its many hundreds of hours of footage highlight its commitment to ensuring these groups are seen onscreen. We commend the NSA for its dedication to centering underrepresented voices, and we hope it serves as an inspiration to others for telling more diverse stories. This is what America looks like.” Nakasone added that he just hoped the NSA’s work would encourage young people from marginalized backgrounds to see themselves and their stories as worthy of constant government surveillance."""
	title = 'Russia wages all-out assault to encircle Ukraine troops in east'

	result = classifier.classify(content)

	print("Our model classifies the news as " + result)