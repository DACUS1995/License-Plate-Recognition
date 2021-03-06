import re
import string
import torch

BATCH_SIZE = 1
OUTPUT_SEQUENCE_LENGTH = 10
ALPHABET = string.ascii_letters + string.digits + "_" #blank char for CTC

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def logc(text, color=bcolors.OKCYAN):
	print(f"{color}{text}{bcolors.ENDC}")

def deEmojify(text):
	regrex_pattern = re.compile(pattern = "["
		u"\U0001F600-\U0001F64F"  # emoticons
		u"\U0001F300-\U0001F5FF"  # symbols & pictographs
		u"\U0001F680-\U0001F6FF"  # transport & map symbols
		u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	"]+", flags = re.UNICODE)

	return regrex_pattern.sub(r'',text)

def tensorToWordSync(tensor):
	indices = torch.argmax(tensor, dim=2).tolist()
	
	words = []
	for batch_idx in range(BATCH_SIZE):
		cur_word_indices = indices[batch_idx]
		cur_word = []
		last_letter = None
		for idx in range(OUTPUT_SEQUENCE_LENGTH):
			cur_word.append(ALPHABET[cur_word_indices[idx]])

		words.append("".join(cur_word))
	return words