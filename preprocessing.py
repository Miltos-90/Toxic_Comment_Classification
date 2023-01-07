""" This module implements the text preprocessing pipeline. 
    It defines all the necessary functions, classes, and objects, and implements all
    the necessary preprocessing steps in the preprocess() function at the bottom.

    Some functions used here come from: https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing
"""

import re
import string
import pkg_resources as pkgr
from num2words import num2words
from symspellpy import SymSpell
from unidecode import unidecode
import contractions
from typing import Union

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

import constants as c

class abbreviations:

    @staticmethod
    def to_text(text:str, abbrvDict:dict = c.ABBREVIATIONS) -> str:
        ''' Converts chat (slang) abbreviatons to standard text. '''

        for abbreviation, meaning in abbrvDict.items(): 

            # Check if abbreviation exists and replace all appearances
            if abbreviation in text: 
                text = re.sub(abbreviation, meaning, text)

            else:

                # Check if it appears lowercased as well
                abbrvLower = abbreviation.lower()
                if abbrvLower in text: 
                    text = re.sub(abbreviation, meaning, text)

        return text

class emojis:

    @staticmethod
    def to_text(text:str, emojiDict:dict = c.EMOJIS) -> str:
        ''' Maps emojis to text 
            
        '''
        
        for emot in emojiDict:
            text = re.sub(u'('+emot+')', emojiDict[emot], text)
        return text

    @staticmethod
    def remove(text:str, pattern = c.PATTERN_EMOJI) -> str:
        ''' Removes Emojis from a string '''
        return pattern.sub(r'', text)

class emoticons:

    @staticmethod
    def remove(text:str, pattern = c.PATTERN_EMOTICON) -> str:
        ''' Removes Emoticons from a string '''
        return pattern.sub(r'', text)

    @staticmethod
    def to_text(text:str, emoticonDict:dict = c.EMOTICONS) -> str:
        ''' Maps emoticons to text '''
        
        for emot in emoticonDict:
            text = re.sub(u'('+emot+')', emoticonDict[emot], text)
            
        return text

class html:

    @staticmethod
    def remove(text:str, pattern = c.PATTERN_HTML) -> str:
        ''' Removes URLs from a string '''
        return pattern.sub(r'', text)

class url:

    @staticmethod
    def remove(text:str, pattern = c.PATTERN_URL) -> str:
        ''' Removes URLs from a string '''
        return pattern.sub(r'', text)

class numeric:

    @staticmethod
    def to_text(text:str) -> str:
        ''' Maps numbers to text (eg 84 -> eighty-four) '''
        
        return re.sub(r'(\d+)', lambda x: num2words(x.group()), text)

class punctuation:

    @staticmethod
    def remove(text:str, pattern = c.PATTERN_PUNCTUATION) -> str:
        ''' Removes punctuation from a string '''
        return pattern.sub(r'', text)

class stopwords:

    @staticmethod    
    def remove(text:str, wordList:set = c.STOPWORDS) -> str:
        ''' Removes stopwords from a string '''
        tokens = [t for t in text.split() if t not in wordList]
        return " ".join(tokens)

class spelling:

    @staticmethod
    def deduplicate(text:str, pattern = c.PATTERN_DUPLICATE_CHARS) -> str:
        """ Removes multiple consecutive sequences of consecutive duplicate characters in a string. 
            eg: cooool -> cool, goooaaal -> goal
        """
        return pattern.sub(r"\1\2", text)

class SpellingCorrector():
    """ Convenience wrapper for spelling correction with SymSpell
        Docs: https://github.com/mammothb/symspellpy
    """
    
    def __init__(self, 
        unigram_txt = "frequency_dictionary_en_82_765.txt",
        bigram_txt  = "frequency_bigramdictionary_en_243_342.txt"):
        """ Initialisation method. Loads the necessary dicts
        """
        
        self.sp = SymSpell(max_dictionary_edit_distance = 2, prefix_length = 7)
        
        # Dict with English words
        dPath = pkgr.resource_filename("symspellpy", unigram_txt)
        self.sp.load_dictionary(dPath, term_index = 0, count_index = 1)
        
        # Path to dict with bigrams
        bPath = pkgr.resource_filename("symspellpy", bigram_txt)
        self.sp.load_bigram_dictionary(bPath, term_index = 0, count_index = 2)
        
        return
    
    
    def __call__(self, text : Union[str, list], max_edit_distance: int = 2, **kwargs):
        """ Corrector for a single word (text of type str) or a list of words
        """
        
        if isinstance(text, list): return [self._correct(w, max_edit_distance, **kwargs) for w in word_tokenize(text)]
        else:                      return  self._correct(text, max_edit_distance, **kwargs)
            

    def _correct(self, text:str, max_edit_distance: int = 2, **kwargs):
        """ Convenience wrapper of the lookup_compound command
        """
        suggestions = self.sp.lookup_compound(text, max_edit_distance, **kwargs)

        return suggestions[0].term

class Lemmatizer():
    ''' Convenience wrapper of the wordnet lemmatizer
    '''

    def __init__(self):

        self.lemmatizer = WordNetLemmatizer()

        return 

    def __call__(self, text: Union[str, list]):
        ''' Lemmatizes a sentence or list of sentences <text> using the wordnet lemmatizer
        '''
        
        if   isinstance(text, list): return [self._lemmatize(sentence) for sentence in text]
        elif isinstance(text, str):  return self._lemmatize(text)
        else: raise TypeError("Only text of type str or list is supported.")


    def _lemmatize(self, text:str):
        ''' Lemmatizes a sentence <text> using the wordnet lemmatizer.
        '''

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

# Instantiate spell correction and lemmatizer objects for function-like use 
lemmatize = Lemmatizer()
correct   = SpellingCorrector()

def preprocess(text:str) -> str:
    ''' Main processing function '''

    text = url.remove(text)                           # Remove URLs
    text = html.remove(text)                          # Remove HTML tags
    text = emoticons.to_text(text)                    # Convert emoticons to words
    text = emojis.to_text(text)                       # Convert emojis to words
    text = abbreviations.to_text(text)                # Convert slang abbreviations
    text = contractions.fix(text)                     # Expand contractions
    text = numeric.to_text(text)                      # Convert digits to words
    text = unidecode(text)                            # Convert accented characters
    text = spelling.deduplicate(text)                 # Remove consecutive multiple instance of duplicated chars 
    text = sent_tokenize(text, language = 'english')  # Split into sentences
    text = lemmatize(text)                            # Lemmatize
    text = [s.lower().strip() for s in text]          # Lower-case and strip leading / trailing spaces
    text = [punctuation.remove(s) for s in text]      # Remove punctuation
    text = [stopwords.remove(s) for s in text]        # Remove stopwords
    text = [correct(s) for s in text]                 # Correct misspelled words
    text = ' '.join(text)                             # Join sentences

    return text