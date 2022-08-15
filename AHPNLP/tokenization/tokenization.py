#####################################
# Functions useful for tokenizing

#####################################


import spacy
import re
from tqdm import tqdm
from unidecode import unidecode # to remove diacritics while filtering

import nltk
from nltk.stem import SnowballStemmer
import os

# get relevant nltk data
try:
    nltk.data.find(os.path.join("tokenizers","punkt"))
    nltk.data.find(os.path.join("corpora", "stopwords"))
except:
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

    def tokenize_standart(
            self, 
            spacy_doc,
            only_nouns = False
            ) -> list:
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
        if only_nouns:
            tokens = [token.lemma_.lower()
                for token in spacy_doc
                if token.pos_ in ["NOUN", "PROPN"]
                ]
        else: 
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

    def clean_parsed_spacy_span(
            self, 
            spacy_spans, 
            document_array_attributes,
            nominal_groups = False
            ):
        """
        input: a span (e.g. from doc.ents or doc.noun_chunks) and 
        their parsing information `document_array_attributes` 

        "clean them out": ignore those with unrecognized POS, those that are too short, etc.
        if `nominal_groups = True`, only keep those with more than one token

        output: list of list of tokens
        """
        noun_POS = ["NOUN", "PROPN"]
        clean_tokens_result = []
        for spacy_span in spacy_spans:
            parsed_span = spacy_span.as_doc(
                array_head = document_array_attributes
                )
            # get all the tokens
    
            tokens = [token 
                for token in parsed_span
            ]

            # don't include noun phrases with tokens without a recognized POS
            has_unrecognized = any([token.pos_ == "X" for token in tokens])
            if has_unrecognized: continue # go to next noun_chunk

            clean_tokens = tokens.copy()
            
            # for nominal groups
            # delete all non-nouns from the beginning of the token list
            if nominal_groups:
                for token in tokens:
                    if (token.pos_ not in noun_POS
                        or 
                        len(token) < 4
                        ):
                        del clean_tokens[0]
                    else:
                        break
            else:
                # delete all non nouns
                tokens = [token for token in tokens if token.pos_ in noun_POS] 
            
            # delete punctuation from the end
            while True:
                try:
                    if clean_tokens[-1].pos_ in ["PUNCT", "SPACE"]: clean_tokens.pop()
                    else: break
                except:
                    break

            # if there is a punctuation in the middle of the token list, delete everything to the right:
            aux_list = []
            for token in clean_tokens:
                if (
                    not all([not char.isalpha() for char in token.text]) # avoid strings that are only punctuation
                    and 
                    token.pos_ not in ["PUNCT", "SPACE"]
                    ): aux_list.append(token)
                else: break
            clean_tokens = aux_list.copy()

            # nominal groups must have at least two tokens (nominal group = noun + complement)
            if nominal_groups:
                if len(clean_tokens) > 1:
                    clean_tokens_result.append(clean_tokens)
            else:
                clean_tokens_result.append(clean_tokens)

            
        return clean_tokens_result
    
        
    
    def get_nominal_groups(self, spacy_doc):
        """given a spacy doc, obtain all its nominal groups"""

        # use the noun chunks, but without determiners
        # also use entities
        
        document_array_attributes = spacy_doc._get_array_attrs()
        
        noun_chunks = list(spacy_doc.noun_chunks) + list(spacy_doc.ents)
        
        nominal_groups = self.clean_parsed_spacy_span(
            noun_chunks, 
            document_array_attributes, 
            nominal_groups=True
            )
        """
        for noun_chunk in noun_chunks:
            parsed_noun_chunk = noun_chunk.as_doc(
                array_head = document_array_attributes
                )
            # get all the tokens
    
            tokens = [token 
                for token in parsed_noun_chunk
            ]

            # don't include noun phrases with tokens without a recognized POS
            has_unrecognized = any([token.pos_ == "X" for token in tokens])
            if has_unrecognized: continue # go to next noun_chunk

            # delete all non-nouns from the beginning of the token list
            clean_tokens = tokens.copy()
            for token in tokens:
                if (token.pos_ not in head_POS
                    or 
                    len(token) < 4
                    ):
                    del clean_tokens[0]
                else:

                    break
            
            # delete punctuation from the end
            while True:
                try:
                    if clean_tokens[-1].pos_ in ["PUNCT", "SPACE"]: clean_tokens.pop()
                    else: break
                except:
                    break

            # if there is a punctuation in the middle of the token list, delete everything to the right:
            aux_list = []
            for token in clean_tokens:
                if (
                    not all([not char.isalpha() for char in token.text]) # avoid strings that are only punctuation
                    and 
                    token.pos_ not in ["PUNCT", "SPACE"]
                    ): aux_list.append(token)
                else: break
            clean_tokens = aux_list.copy()

            # nominal groups must have at least two tokens (nominal group = noun + complement)
            if len(clean_tokens) > 1:
                nominal_groups.append(clean_tokens)

                
        # add entities
        #nominal_groups.extend( list(spacy_doc.ents) )
        """
        return nominal_groups
    
    def tokenize_for_nominal_groups(self, spacy_doc) -> list:
        """given a spacy doc, get a list of all its nominal groups"""

        nominal_groups = self.get_nominal_groups(spacy_doc)
        nominal_groups = [quick_clean_string(
            #noun_chunk.text
            (
                " ".join([token.text for token in noun_chunk])
                ).replace("’ ","’") # "d' honneur" -> "d'honneur"
            ) for noun_chunk in nominal_groups
            ]
        nominal_groups = self.clean_tokens_list(
            nominal_groups, strong_filter = False
            ) # include non alphanumeric characters
        
    
        return nominal_groups
    
    def tokenize_search_words(self, term_list:list) -> list:
        """
        Given a list of terms searched, return the list of lists of all the tokens and entities in those search terms
        e.g. 
        input: "fonctions fuchsiennes", "nobel", "marie et pierre" 
        output: [ [fonction, fuchsien], [nobel], [marie, pierre] ]
        """
        
        result = []
        for search_term in term_list:
            spacy_doc = self.spacy_pipeline(search_term)
            tokens = self.tokenize_standart(spacy_doc) #+ self.get_tokenized_entities(spacy_doc)
            tokens = self.clean_tokens_list(tokens)
            #tokens = list(set(tokens))
            result.append( tokens )
            
        return result


        # Legacy
        """ 
        separator = ", "
        text_string = separator.join(term_list)
        spacy_doc = self.spacy_pipeline(text_string)

        tokens = self.tokenize_standart(spacy_doc)
        entities = [quick_clean_string(ent.text) for ent in spacy_doc.ents]

        result = tokens + entities
        result = self.clean_tokens_list(result)

        return result
        """

    def get_entities(self, spacy_doc):
        """get a list of entities in the document"""
        
        entities = spacy_doc.ents
        document_array_attributes = spacy_doc._get_array_attrs()
        entities = self.clean_parsed_spacy_span(
            entities, 
            document_array_attributes,
            )
        return entities
    
    def get_tokenized_entities(self, spacy_doc):
        """get a list ot list of tokens from the entities in the document"""
        tokenized_entities = self.get_entities(spacy_doc)
        res = []
        for token_list in tokenized_entities:
            res.extend(token_list)
        
        res = [token.text for token in res]
        return res


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

