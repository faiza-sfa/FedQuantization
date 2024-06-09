### Origin of code: LEAF

import os
import re
from src import LEAF_PATH
import json
import torch

# ------------------------
# utils for sent140 dataset

def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split
    
    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices
    
    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer 
    representing unknown index to returned list until the list's length is 
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl

class EmbeddingTransformer():
    # Make the embedding class variables, because they take up a lot of memory
    # and have no instance-specific state.
    is_loaded = False
    def __init__(self):
        if not EmbeddingTransformer.is_loaded:
            EmbeddingTransformer.is_loaded = True
            print("Loading embs.json")
            with open(os.path.join(LEAF_PATH, "models", "sent140", "embs.json")) as inf:
                embs = json.load(inf)
            print("Done loading embs.json")
            id2word = embs['vocab']
            EmbeddingTransformer.word2id = {v: k for k,v in enumerate(id2word)}
            EmbeddingTransformer.word_emb = torch.nn.Embedding.from_pretrained(torch.tensor(embs['emba']))
            del embs

    def __call__(self, x, y):
        max_words = 25
        x_idxs = torch.tensor(line_to_indices(x[4], EmbeddingTransformer.word2id, max_words))
        y_label = torch.tensor(int(y))
        x_emb = EmbeddingTransformer.word_emb(x_idxs)
        return x_emb, y_label

# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_idx(letter):
    return ALL_LETTERS.find(letter)


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

class EmbeddingTransformerShakespeare():
    def __call__(self, x, y):
        x_idxs = torch.tensor(word_to_indices(x))
        y_label = torch.tensor(letter_to_idx(y))
        return x_idxs, y_label
