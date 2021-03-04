import torch
import numpy as np

from train import ALPHABET, BATCH_SIZE, OUTPUT_SEQUENCE_LENGTH


def tensorToWord(tensor):
	tensor = tensor.permute([1, 0, 2])
	indices = torch.argmax(tensor, dim=2).tolist()
	
	words = []
	for batch_idx in range(BATCH_SIZE):
		cur_word_indices = indices[batch_idx]
		cur_word = []
		last_letter = None
		for idx in range(OUTPUT_SEQUENCE_LENGTH):
			if ALPHABET[cur_word_indices[idx]] == ALPHABET[-1]:
				last_letter = None
				continue
			else:
				if last_letter == None or (last_letter is not None and last_letter != cur_word_indices[idx]):
					cur_word.append(ALPHABET[cur_word_indices[idx]])
					last_letter = cur_word_indices[idx]
					continue
		words.append("".join(cur_word))
	return words


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


def levenshteinDistance(string_one, string_two):
    dist = np.zeros((len(string_one) + 1, len(string_two) + 1))
    dist[1:, 0] = [i + 1 for i in range(len(string_one))]
    dist[0, 1:] = [i + 1 for i in range(len(string_two))]

    for row in range(1, len(string_one) + 1):
        for col in range(1, len(string_two) + 1):
            gain = 1
            if string_one[row - 1] == string_two[col -1]:
                dist[row, col] = dist[row - 1, col -1]
            else:
                dist[row, col] = gain + min(dist[row - 1, col], dist[row, col - 1], dist[row - 1, col - 1])
    
    return dist[-1,-1]


def batchLevenshteinDistance(arr_one, arr_two):
    cumulative_distance = 0
    for idx in range(len(arr_one)):
        cumulative_distance += levenshteinDistance(arr_one[idx], arr_two[idx])
    return cumulative_distance
