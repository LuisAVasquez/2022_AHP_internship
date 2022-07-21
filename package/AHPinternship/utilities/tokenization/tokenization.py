#####################################
# Functions useful for tokenizing

#####################################


import spacy
import re
from tqdm import tqdm
from unidecode import unidecode # to remove diacritics while filtering

class Tokenizer:
    
    def __init__(self, language = None):
        """
        arguments
        `language` is the common name of the langage. e.g. "english", not "EN".
        By default, use "french".
        """
        # default to french
        if language is None: language = "french"

        # load spacy pipeline
        
        if language == "french" : self.spacy_pipeline = spacy.load('fr_core_news_sm')
        if language == "english": self.spacy_pipeline = spacy.load('en_core_web_sm')
        if language == "german" : self.spacy_pipeline = spacy.load('de_core_news_sm')
        if language == "italian": self.spacy_pipeline = spacy.load('it_core_news_sm')
        if language == "swedish": self.spacy_pipeline = spacy.load('sv_core_news_sm')

        # get stopwords
        self.stopwords = self.spacy_pipeline.Defaults.stop_words


        return None

    def tokenize(self, text_string : str) -> list:
        """Given a string, return its list of tokens"""

        # Part Of Speech to ignore:
        undesired_POS = [
            #'ADV',
            'PRON',
            'CCONJ',
            'PUNCT',
            'PART',
            'DET',
            'ADP', 
            'SPACE'
            ]
        #create spacy document
        spacy_doc = self.spacy_pipeline(text_string)

        tokens = [
            token.lemma_.lower() #lowercase lemma
            for token in spacy_doc 
            if token.pos_ not in undesired_POS
        ]

        pattern = "^[a-zA-Z]+$"

        # only keep tokens made up of latin-like characters
        tokens = [
            token 
            for token in tokens 
            if ( re.match(pattern , unidecode(token) )  # string without diacritics
                and len(token) > 3 # length at least 4
                )
            ]
        #filter stopwords
        tokens = [
            token 
            for token in tokens 
            if token not in self.stopwords
            ]
        return tokens

    def __call__(self, text_string : str) -> list:
        return self.tokenize(text_string)

    def batch_tokenize(self, string_list:list) -> list:
        """
        Tokenize a batch of texts
        """
        res = []
        for document in tqdm(list(string_list)):
            res.append(self.tokenize(document))
    
        return res

