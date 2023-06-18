import numpy as np
from . import config
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import text, sequence
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist


def split(X: np.array, y: np.array, 
          shuffle   = True,
          trainSize = config.TRAIN_PERCENTAGE,
          valSize   = config.VAL_PERCENTAGE,
          seed      = config.RANDOM_SEED):
    """ Splits data into train/validation/test sets. """

    Xl, Xvt, yl, yvt = train_test_split(X, y,     train_size = trainSize,                 shuffle = shuffle, random_state = seed, stratify = y)
    Xv, Xt, yv, yt   = train_test_split(Xvt, yvt, train_size = valSize / (1 - trainSize), shuffle = shuffle, random_state = seed, stratify = yvt)

    return Xl, Xv, Xt, yl, yv, yt


def countWords(corpus: np.array) -> dict:
    """ Counts words in a list of texts """

    fDist = FreqDist()

    for text in tqdm(corpus):
        words = word_tokenize(text)
        
        for word in words:
            fDist[word] += 1

    return dict(fDist)


def filterWords(corpus: np.array, wordDict:set) -> np.array:
    """ Removes all words apart from the words appearing in wordDict
        from a corpus (an iterable of strings) 
    """

    corpusFiltered = np.empty_like(corpus)
    iterable       = tqdm(enumerate(corpus), total = corpus.shape[0])

    for idx, text in iterable:
        wordsRaw            = word_tokenize(text)
        wordsFiltered       = [w for w in wordsRaw if w in wordDict]
        corpusFiltered[idx] = ' '.join(wordsFiltered)

    return corpusFiltered


def embeddingOutputSize(numCategories: int) -> int:
    """ Empirical rule for the calculation of the embedding size based on the number of categories, as discussed in [1].
        The rule for the small number of categories (<=1000) is also recommended in [2]. 
        For a large number of categories (>1000) the rule recommended in [2, 3] is employed

        [1] https://ai.stackexchange.com/questions/28564/how-to-determine-the-embedding-size?newreg=8aa67c28d4e241b6a8e47360cbf1f827.
        [2] https://forums.fast.ai/t/size-of-embedding-for-categorical-variables/42608/2?u=ste.
        [3] Lakshmanan, Valliappa, Sara Robinson, and Michael Munn. Machine learning design patterns. O'Reilly Media, 2020.
    """

    if numCategories <= 1000:
        return min(50, numCategories // 2 ) # [1, 2]
    else:
        return min(600, round(1.6 * numCategories ** 0.56) ) # [2, 3]


class Vectorizer:
    """ Convenience class to convert text to sequences """

    def __init__(self, 
                 vocabularySize: int = None, # Number of most frequent qords to retain in the vocabulary
                 sequenceLength: int = None  # Maximum sequence length. Longer sequences are cut, shorter sequences are padded
                 ):
        """ Initialisation method """

        self.tokenizer = text.Tokenizer(num_words = vocabularySize, lower = False)
        self.seqLength = sequenceLength

        return 

    """ Fit methods for the tokenizer """
    def fitOnTexts(self, X:np.array):     self.tokenizer.fit_on_texts(X)
    def fitOnSequences(self, X:np.array): self.tokenizer.fit_on_sequences(X)

    
    def textsToSequences(self, X:np.array) -> np.array:
        """ Converts texts to sequences """

        X = self.tokenizer.texts_to_sequences(X)
        
        if self.seqLength is not None: 
            X = sequence.pad_sequences(X, maxlen = self.seqLength)

        return X

    def getVocabulary(self) -> dict:
        """ Returns the vocabulary of the dataset """

        return self.tokenizer.word_index

