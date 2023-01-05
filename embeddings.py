""" This module contains all necessary functions required to generate the embedding matrix. """

import numpy as np
import io


def _readFile(pathToFile     : str,
              vocabularyDict : dict, 
              encoding       : str  = 'utf-8',
              newlineChar    : str  = '\n',
              errors         : str  = 'ignore',
              verbose        : bool = True) -> dict:
    """ Reads the file containing the word embeddings, in which each line corresponds
        to a single word. Expected format for each line:
        <word> <embeddingValueDim1> <embeddingValueDim2> ... <embeddingValueDimN><newlineChars>
        Returns a dictionary whose keys correspond to words and the embeddings to their values.
    """

    if verbose: print(f'Loading {pathToFile}...')
    data = {} # Dict to hold embeddings for each word

    with io.open(pathToFile, 'r', encoding = encoding, newline = newlineChar, errors = errors) as f:

        # Read each line and populate the dictionary
        for line in f:
            tokens  = line.rstrip().split(' ')
            word    = tokens[0] 
            vector  = np.asarray(tokens[1:], dtype = 'float32')
            wordIdx = vocabularyDict.get(word)
            if wordIdx: data[wordIdx] = vector

    return data


def _makeMatrix(embeddingDict  : dict, 
                vocabularyDict : dict, 
                embeddingDim   : int,
                verbose        : bool = True) -> np.array:
    """ Makes the embedding matrix (vocabulary size x embeddingDim), given an embedding dictionary
        (keys: words, values: embeddings) and a vocabulary dictionary (keys: words, values: word indices).
    """

    hits, misses = 0, 0                               # Counters for num words converted
    noTokens     = len(vocabularyDict) + 1            # +1 for OOV tokens
    matrix       = np.zeros((noTokens, embeddingDim)) # Embedding matrix

    # Loop through the vocabulary and make the embedding matrix
    if verbose: print("Generating matrix...")

    for word, i in vocabularyDict.items():
        
        embeddingVector = embeddingDict.get(i)
        if embeddingVector is not None:
            matrix[i] = embeddingVector
            hits += 1
        else:
            misses += 1

    if verbose: 
        print("Converted %d words (%d misses)" % (hits, misses))
        print(f'Embedding matrix shape: {matrix.shape}')

    return matrix


def generate(embeddingFile  : str, 
             vocabularyDict : dict, 
             embeddingDim   : int) -> np.array:
    """ Generates the embedding matrix (vocabulary size x embeddingDim).
        Assumes that words in the vocabulary are all lower-cased.
        Embedding for Out-of-vocab (OOV) tokens is all-zeroes.
    """

    embeddingDict   = _readFile(embeddingFile, vocabularyDict)
    embeddingMatrix = _makeMatrix(embeddingDict, vocabularyDict, embeddingDim)

    return embeddingMatrix