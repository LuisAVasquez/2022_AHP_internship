##########################
# utilities for topic modelling
##########################

from tqdm import tqdm
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from AHPtopicmodelling.utilities.tokenization import tokenization
from AHPtopicmodelling.utilities import so
# default maximum quantity of topic words to get after training the topic model.
MAX_TOPIC_TOKENS = 4 

# default threshold for discarding tokens. All tokens above this percentage of documents will be considered uninformative
# Motivation: if all documents start with "hello!", then this token is not useful to determine a topic.
DEFAULT_NO_ABOVE = 0.75 

class TopicModelling():
    """
    Class for topic modelling on documents
    """
    def __init__(self, 
        documents:list,  # list of strings
        language = None, 
        no_above = None
        ):
        """
        `documents` is  a list of strings
        `language` is the language of the documents. default: "french"
        """

        if language is None: language = "french"
        self.language = language
        self.documents = documents

        # tokenize the documents
        self.tokenizer = tokenization.Tokenizer(self.language)
        self.tokenized_documents = self.tokenizer.batch_tokenize(documents)

        # other attributes
        self.no_above = no_above    # upper threshold for documents. float between 0 and 1
        self.dictionary = None  # gensim dictionary
        self.corpus = None      # gensim corpus
        self.model = None       # gensim model
        self.coherence_cv = None# model coherence
        self.topics_for_all_documents = None # topics for all documents
        return None

    def get_gensim_dictionary(
        self,
        no_above = DEFAULT_NO_ABOVE,
        debug = False
    ):
        """
        get a gensim id-word dictionary
        """
        
        if self.dictionary is not None: return self.dictionary # avoid re-computation
        
        documents = self.tokenized_documents
        dictionary = corpora.Dictionary(documents)
        if debug: print("Unique tokens before filtering:", len(dictionary))
        
        if len(documents) > 10: # only filter if there are more than 10 documents
            dictionary.filter_extremes(
                no_below = 5, 
                no_above = no_above #0.75#0.25
            )
            if debug: print('Unique tokens after filtering:', len(dictionary))
        else: 
            if debug: print("not enough documents to justify filtering")
            if debug: print("Unique tokens", len(dictionary))

        # store the dictionary
        self.no_above = no_above
        self.dictionary = dictionary
        return dictionary
    
    def get_gensim_corpus(self, no_above):
        """
        get a list of documents as Bag-of-Words lists.
        """
        if self.corpus is not None: return self.corpus
        dictionary = self.get_gensim_dictionary(no_above)
        documents = self.tokenized_documents
    
        corpus = [ dictionary.doc2bow(doc) for doc in documents]
        
        #store the corpus
        self.corpus = corpus
        return corpus

    def get_gensim_model(
        self,
        num_topics,
        alpha,# dirichlet hyperparameter alpha: Document-Topic Density
        eta, # Dirichlet hyperparameter eta: Word-Topic Density
        no_above
        ):
        # load dictionary and corpus
        dictionary = self.get_gensim_dictionary(no_above)
        corpus = self.get_gensim_corpus(no_above)

        # train the topic model
        lda_model = gensim.models.LdaMulticore(
            corpus = corpus, 
            id2word = dictionary, 
            num_topics = num_topics, 
            chunksize = 100, 
            passes = 10, 
            random_state=1,
            alpha = alpha,
            eta = eta
        ) 
        # store the model
        self.model = lda_model
        return lda_model

    def check_if_model_trained(self):
        if self.model is None:
            raise ValueError("You first need to compute the gensim model! Use .get_gensim_model()")
        pass

    def get_coherence_score(self):
        
        self.check_if_model_trained()
        if self.coherence_cv is not None: return self.coherence_cv
        
        gensim_model = self.model
        gensim_dictionary = self.dictionary
        tokenized_documents = self.tokenized_documents

        coherence_model_lda = CoherenceModel(
            model = gensim_model, 
            texts =  tokenized_documents,
            dictionary = gensim_dictionary, 
            coherence = 'c_v'
        )
        self.coherence_cv = coherence_model_lda.get_coherence()
        return self.coherence_cv


    def visualize_topics(self):
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis
        pyLDAvis.enable_notebook()
        
        self.check_if_model_trained()
        lda_model = self.model

        dictionary = self.get_gensim_dictionary(self.no_above)        
            
        corpus = self.get_gensim_corpus(self.no_above)

        vis = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds')
        return vis

    def get_topic_words_for_tokenized_document(
        self,
        tokenized_document,
        max_topics = MAX_TOPIC_TOKENS
        ):
        """
        get keywords for a tokenized document (a list of tokens), 
        using a lda model
            
        """
        self.check_if_model_trained()
        lda_model = self.model
        dictionary = self.dictionary

        corpus = dictionary.doc2bow(tokenized_document)

        # get list of (topic_index, prob)
        document_topics = lda_model.get_document_topics(
            corpus
        )

        document_topics = sort_by_probability(document_topics)

        # get list to (word, prob)
        document_topics = [(dictionary[id],prob) for id, prob in document_topics]

        # how many words to get?
        max_topics = min(max_topics, len(document_topics))
        result = document_topics[:max_topics]

        result = [word for word, prob in result]

        return result

    def get_topic_words_for_new_document(
        self, 
        text_string, 
        max_topics = MAX_TOPIC_TOKENS
        ):
        """Just tokenize a document and get its topics"""
        
        tokenized_document = self.tokenizer.tokenize(text_string)
        return self.get_topic_words_for_tokenized_document(
            tokenized_document,
            max_topics = max_topics
        )
        
    def get_topics_for_all_documents(
        self,
        max_topics = MAX_TOPIC_TOKENS
        ):
        self.check_if_model_trained()
        #if self.topics_for_all_documents is not None: return self.topics_for_all_documents
        
        topics_for_all_documents = []

        for tokenized_document in tqdm(self.tokenized_documents):

            topics_for_document = self.get_topic_words_for_tokenized_document(
                lda_model = self.model,
                tokenized_document= tokenized_document,
                dictionary= self.dictionary,
                max_topics= max_topics
            )
            topics_for_all_documents.append(topics_for_document)

        self.topics_for_all_documents = topics_for_all_documents
        return topics_for_all_documents



