#####################################
# Functions useful for tokenizing

#####################################


import spacy
import re
from tqdm import tqdm
from unidecode import unidecode # to remove diacritics while filtering

import nltk
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')

from AHPNLP.utilities.utilities import quick_clean_string

class Tokenizer:
    
    def __init__(self, 
        language = None,
        nominal_groups = False,
        ):
        """
        arguments
        `language` is the common name of the langage. e.g. "english", not "EN".
        By default, use "french".
        If `nominal_groups` is true, extract the noun phrases from the text string. Otherwise just tokenize
        """
        # default to french
        if language is None: language = "french"

        # load spacy pipeline
        
        if language == "french" : self.spacy_pipeline = spacy.load('fr_core_news_sm')
        if language == "english": self.spacy_pipeline = spacy.load('en_core_web_sm')
        if language == "german" : self.spacy_pipeline = spacy.load('de_core_news_sm')
        if language == "italian": self.spacy_pipeline = spacy.load('it_core_news_sm')
        if language == "swedish": self.spacy_pipeline = spacy.load('sv_core_news_sm')
        self.language = language
        self.stemmer = SnowballStemmer(language)

        # get stopwords
        self.stopwords = self.spacy_pipeline.Defaults.stop_words

        # nominal groups flag
        self.nominal_groups = nominal_groups
        
        # current spacy doc
        self.current_doc = None

        return None

    def clean_tokens_list(
        self, 
        string_list: list,
        strong_filter = True, #if True, exclude non-alphanumeric characters
        ) -> list:
        """given a list of tokens, normalize them."""

        # the transcriptions come to the footnote numbers added to the end of the words in which they appear.
        # these removes those numbers
        tokens = [ quick_clean_string(string) for string in string_list]

        pattern = "^[a-zA-Z]+$"
        
        
        # only keep tokens made up of latin-like characters
        if strong_filter:
            tokens = [token for token in tokens
                if re.match(pattern, unidecode(token)) # string without diacritics and punctuation
            ]
        # and of length at least 4
        tokens = [
            token 
            for token in tokens 
            if len(token) > 3 # length at least 4
            ]

        #filter stopwords
        tokens = [
            token 
            for token in tokens 
            if token not in self.stopwords
            ]
        return tokens

    def tokenize_standart(self, spacy_doc) -> list:
        """Given a spacy doc, return its list of tokens"""
        
        # Part Of Speech to ignore:
        undesired_POS = [
            #'ADV',
            #'PRON',
            'CCONJ',
            'PUNCT',
            'PART',
            'DET',
            'ADP', 
            'SPACE'
            ]
        
        tokens = [
            token.lemma_.lower() #lowercase lemma
            for token in spacy_doc 
            if token.pos_ not in undesired_POS
        ]

        tokens = self.clean_tokens_list(tokens)

        # tokens = [
        #     self.stemmer.stem(token )
        #     for token in tokens
        # ]

        return tokens
    
    def get_nominal_groups(self, spacy_doc):
        """given a spacy doc, obtain all its nominal groups"""

        # just using the noun chunks deteted by spacy
        nominal_groups = list(spacy_doc.noun_chunks) + list(spacy_doc.ents)

        return nominal_groups
    
    def tokenize_for_nominal_groups(self, spacy_doc) -> list:
        """given a spacy doc, get a list of all its nominal groups"""

        nominal_groups = self.get_nominal_groups(spacy_doc)
        
        nominal_groups = [quick_clean_string(noun_chunk.text) for noun_chunk in nominal_groups]
        nominal_groups = self.clean_tokens_list(
            nominal_groups, strong_filter = False
            ) # include non alphanumeric characters
        
    
        return nominal_groups
    
    def tokenize_search_words(self, term_list:list) -> list:
        """
        Given a list of terms searched, return the list of all the tokens and entities in those search terms
        """
        separator = ", "
        text_string = separator.join(term_list)
        spacy_doc = self.spacy_pipeline(text_string)

        tokens = self.tokenize_standart(spacy_doc)
        entities = [quick_clean_string(ent.text) for ent in spacy_doc.ents]

        result = tokens + entities
        result = self.clean_tokens_list(result)

        return result



    def tokenize(self, text_string : str) -> list:
        """Given a string, return its list of tokens, depending on whether or not we are running
        standard tokenization or nominal group extraction
        """
        # initialize the current spacy document
        self.current_doc = self.spacy_pipeline(text_string)
        if self.nominal_groups:
            return self.tokenize_for_nominal_groups(self.current_doc)
        else:
            return self.tokenize_standart(self.current_doc)

    def get_nominal_groups_tokens_dictionary(self, spacy_doc = None):
        """
        get a dictionary of the form
        "nominal group text" : [lemmatized tokens]
        """
        if not self.nominal_groups: raise ValueError("Initialize the tokenizer with nominal_groups = True")

        if spacy_doc is None: spacy_doc = self.current_doc

        nominal_groups_tokens_dictionary = {}
        nominal_groups = self.get_nominal_groups(spacy_doc)
        document_array_attributes = spacy_doc._get_array_attrs()

        for nominal_group in nominal_groups:

            nominal_group_key = quick_clean_string(nominal_group.text)

            # treat the noun chunk as a spacy doc
            nominal_group_tokens = self.tokenize_standart(
                nominal_group.as_doc(
                array_head = document_array_attributes
                )
            )
            # only add those noun chunks with relevant tokens: e.g. don't include "je"
            if len(nominal_group_tokens) > 0:
                nominal_groups_tokens_dictionary[nominal_group_key] = nominal_group_tokens
        
        return nominal_groups_tokens_dictionary


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

