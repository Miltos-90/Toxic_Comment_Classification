""" This module implements the text preprocessing pipeline. 
    It defines all the necessary functions, classes, and objects, and implements all
    the necessary preprocessing steps in the preprocess() function at the bottom.
"""

import re
import pkg_resources as pkgr
from symspellpy import SymSpell
from unidecode import unidecode
import contractions
from typing import Union
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from . import constants as c


class regexProcessor(object):
    ''' Generic class to replace or remove a set of regex patterns'''

    def __init__(self,
        rDict : dict,            # dictionary containing (keys: regexes, values: replacement texts)
        flags : re.RegexFlag = 0 # ASCII flags for compilation (https://docs.python.org/3/library/re.html#re.ASCII)
    ):
        ''' Compiles the keys of the input dictionary containing the abbreviations and their replacement text.'''

        self.dict = defaultdict()

        for key, value in rDict.items():
            self.dict[re.compile(key, flags = flags)] = value
        
        return

    def transform(self, text:str) -> str:
        ''' Converts all patterns(keys) to their corresponding values '''
    
        for key, value in self.dict.items():
            text = key.sub(value, text)
        
        return text


class SpellingCorrector():
    """ Convenience wrapper for spelling correction with SymSpell
        Docs: https://github.com/mammothb/symspellpy
    """
    
    def __init__(self, 
        unigram_txt = "frequency_dictionary_en_82_765.txt",
        bigram_txt  = "frequency_bigramdictionary_en_243_342.txt"
    ):
        """ Initialisation method. Loads the necessary dicts """
        
        self.sp = SymSpell(max_dictionary_edit_distance = 2, prefix_length = 7)
        
        # Dict with English words
        dPath = pkgr.resource_filename("symspellpy", unigram_txt)
        self.sp.load_dictionary(dPath, term_index = 0, count_index = 1)
        
        # Path to dict with bigrams
        bPath = pkgr.resource_filename("symspellpy", bigram_txt)
        self.sp.load_bigram_dictionary(bPath, term_index = 0, count_index = 2)
        
        return
    
    
    def __call__(self, text : Union[str, list], max_edit_distance: int = 2, **kwargs):
        """ Corrector for a single word (text of type str) or a list of words """
        
        if isinstance(text, list): 
            return [self._correct(w, max_edit_distance, **kwargs) for w in word_tokenize(text)]

        else:                      
            return self._correct(text, max_edit_distance, **kwargs)
            

    def _correct(self, text:str, max_edit_distance: int = 2, **kwargs):
        """ Convenience wrapper of the lookup_compound command """

        suggestions = self.sp.lookup_compound(text, max_edit_distance, **kwargs)

        return suggestions[0].term


class Lemmatizer():
    ''' Convenience wrapper of the wordnet lemmatizer '''

    def __init__(self):

        self.lemmatizer = WordNetLemmatizer()

        return 

    def __call__(self, text: Union[str, list]):
        ''' Lemmatizes a sentence or list of sentences <text> using the wordnet lemmatizer '''
        
        if   isinstance(text, list): return [self._lemmatize(sentence) for sentence in text]
        elif isinstance(text, str):  return self._lemmatize(text)
        else:                        raise TypeError("Only text of type str or list is supported.")


    def _lemmatize(self, text:str):
        ''' Lemmatizes a sentence <text> using the wordnet lemmatizer. '''

        tokens = word_tokenize(text)
        tags   = [(word, self._penn2morphy(tag)) for word, tag in pos_tag(tokens)]
        lemmas = [word if tag is None else self.lemmatizer.lemmatize(word, tag) 
                  for word, tag in tags]
        
        return " ".join(lemmas)


    @staticmethod
    def _penn2morphy(treebankTag:str) -> str:
        ''' Maps Treebank tags to WordNet part of speech names 
            Source: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python/15590384#15590384
        '''
        
        if   treebankTag.startswith('J'):  return wordnet.ADJ
        elif treebankTag.startswith('N'):  return wordnet.NOUN
        elif treebankTag.startswith('R'):  return wordnet.ADV
        elif treebankTag.startswith('M') : return wordnet.VERB
        elif treebankTag.startswith('V'):  return wordnet.VERB
        else:                              return None


""" Instantiate all objects once upon calling the module """
html          = regexProcessor(rDict = c.HTML,          flags = re.I)
url           = regexProcessor(rDict = c.URL_STRIP,     flags = re.I)
abbreviations = regexProcessor(rDict = c.ABBREVIATIONS, flags = re.I)
emoticons     = regexProcessor(rDict = c.EMOTICONS,     flags = re.UNICODE)
emojis        = regexProcessor(rDict = c.EMOJIS,        flags = re.UNICODE)
numeric       = regexProcessor(rDict = c.NUMERIC)
punctuation   = regexProcessor(rDict = c.PUNCTUATION)
stopwords     = regexProcessor(rDict = c.STOPWORDS,     flags = re.I)
duplicates    = regexProcessor(rDict = c.DUPE_CHARS,    flags = re.I)
lettercase    = regexProcessor(rDict = c.LETTERCASE)
multispace    = regexProcessor(rDict = c.MULTISPACE)
lemmatize     = Lemmatizer()
correct       = SpellingCorrector()


def preprocess(text:str) -> str:
    ''' Main processing function '''

    text = html.transform(text)                      # Remove HTML tags
    text = url.transform(text)                       # Remove URLs
    text = emoticons.transform(text)                 # Convert emoticons to words
    text = emojis.transform(text)                    # Convert emojis to words
    text = abbreviations.transform(text)             # Convert slang abbreviations
    text = contractions.fix(text)                    # Expand contractions
    text = numeric.transform(text)                   # Convert digits to words
    text = unidecode(text)                           # Convert accented characters
    text = lettercase.transform(text)                # Standardize upper- and lower-case characters
    text = duplicates.transform(text)                # Remove consecutive multiple instance of duplicated chars 
    text = sent_tokenize(text, language = 'english') # Split into sentences
    text = lemmatize(text)                           # Lemmatize
    text = [s.lower() for s in text]                 # Lower-case
    text = [punctuation.transform(s) for s in text]  # Remove punctuation
    text = [correct(s) for s in text]                # Correct misspelled words
    text = [stopwords.transform(s) for s in text]    # Remove stopwords
    text = ' '.join(text)                            # Join sentences 
    text = multispace.transform(text)                # Remove multiple consecutive spaces
    return text.strip()                              # Remove leading/trailing whitespace
