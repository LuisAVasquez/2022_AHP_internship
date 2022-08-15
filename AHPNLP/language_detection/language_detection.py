

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0 #make the result deterministic

def try_detect(string):
    """try to detect the language of a string"""

    if not isinstance(string, str):
        print("careful with: ", string)
        string = str(string)
    try:
        res = detect(string)
    except:
        res = None
    return res

from numpy import NaN
def get_language_name(abbreviation):
    """
    Given the abbreviation for a language used in the corpus, return the full name
    """
    if abbreviation == 'de':
        return 'german'
    elif abbreviation == 'en':
        return 'english'
    elif abbreviation == 'fr':
        return 'french'
    elif abbreviation == 'it':
        return 'italian'
    elif abbreviation == 'sv' or abbreviation == 'se':
        return 'swedish'
    else:
        print("Could not recognize ", abbreviation)
        return NaN

from nltk import word_tokenize

def language_detection_for_dataframe(
    df,
    column_name = 'clean_transcription'
    ):
    """expects a pandas Dataframe with a 'language' column
    and the text in the `column`
    """
    # determine the language to use
    # 1. for each letter, use the language from the database, 
    # 2. If the language is not available, detect it automatically

    languages_for_preprocessing = []

    for ind, row in df.iterrows():
        db_language = row['language']
        
        if isinstance(db_language, str): 
            # the language is available in the database
            language = get_language_name(db_language)
            
        
        else:
            transcription = row[column_name]

            # delete tokens at the extremes of the transcription, as they may have words in 
            # languages different than the body
            quick_tokenization = word_tokenize(transcription)
            quick_tokenization = quick_tokenization[10:]
            quick_tokenization = quick_tokenization[:-10]
            effective_transcription = " ".join(quick_tokenization)
            
            detected_lang = try_detect(effective_transcription)
            language = get_language_name(detected_lang)
        
        languages_for_preprocessing.append(language)

    # store in the dataframe
    df['language_for_preprocessing'] = languages_for_preprocessing
    df = df.dropna(subset=['language_for_preprocessing'])

    return df
