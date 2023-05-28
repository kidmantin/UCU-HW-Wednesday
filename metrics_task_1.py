import numpy as np
import torch
import re

def levenstein(str_a, str_b) -> int:
    m = len(str_a)
    n = len(str_b)

    matrix = np.zeros((m+1, n+1), dtype=int)

    # Adding first row and column of matrix
    matrix[0] = np.arange(n+1)
    matrix[:, 0] = np.arange(m+1)

    # Filling the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str_a[i-1] == str_b[j-1]:
                matrix[i, j] = matrix[i-1, j-1]
            else:
                matrix[i, j] = min(
                    matrix[i-1, j] + 1,      # Deletion
                    matrix[i, j-1] + 1,      # Insertion
                    matrix[i-1, j-1] + 1     # Substitution
                )

    return matrix[m, n]

def cer(str_a , str_b) -> float:
    '''str_a - predicted, str_b - reference'''
    return levenstein(str_a, str_b)/len(str_b)

def wer(str_a, str_b) -> float:
    '''str_a - predicted sentence, str_b - reference sentence
    punctuation is removed automatically'''
    # Remove punctuation using regular expressions
    str_a_token = [x.lower() for x in re.sub(r'[^\w\s]', '', str_a).split()]
    str_b_token = [x.lower() for x in re.sub(r'[^\w\s]', '', str_b).split()]
    
    return levenstein(str_a_token, str_b_token)/len(str_b_token)

def levenshtein_torch(tokens, targets):
    m = tokens.size(0)
    n = targets.size(0)

    distances = torch.zeros((m + 1, n + 1), dtype=torch.long)

    for i in range(m + 1):
        distances[i][0] = i

    for j in range(n + 1):
        distances[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens[i - 1] == targets[j - 1]:
                distances[i][j] = distances[i - 1][j - 1]
            else:
                deletion = distances[i - 1][j] + 1
                insertion = distances[i][j - 1] + 1
                substitution = distances[i - 1][j - 1] + 1
                distances[i][j] = min(deletion, insertion, substitution)

    return distances[m][n]

def wer_torch(pred_batch, true_batch):
    error_rate = 0
    for tokens, targets in zip(pred_batch, true_batch):
        distance = levenshtein_torch(tokens, targets)
        max_len = max(len(tokens), len(targets))
        error_rate += distance / max_len
    return error_rate/pred_batch.shape[0]


if __name__ == "__main__":
    print(f"words: banana, bobana: cer - {cer('banana', 'bobana')}") # returns 0.3333333333333333, 2 (different characters) / 6 (number of characters)
    print(f"Sentence 1: Joyce enjoyed eating pancakes with ketchup. \n \
        Sentence 2: Rick enjoyed eating salad with ketchup. \n \
        wer - {wer('Joyce enjoyed eating pancakes with ketchup.', 'Rick enjoyed eating salad with ketchup.')}") # returns 0.3333333333333333, 2 (different words) / 6 (number of words) = 0.3333333333333333

