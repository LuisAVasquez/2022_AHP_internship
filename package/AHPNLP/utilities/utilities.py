
from numpy import array
import re

def sort_by_probability(pair_list):
    """
    Orders lists with elements of the form (<something>, probability)
    in descending order by probability
    """
    pair_list = list(pair_list)
    probabilities = [prob for _, prob in pair_list]
    
    # sort in descending order
    indexes_by_probability = array(probabilities).argsort()[::-1]
    result = [pair_list[i] for i in indexes_by_probability]
    return result


def get_clean_transcription(text:str):
    """delete the "Apparat critique", "Notes", and "References" sections"""
    text = text.replace("↩", " ")
    delimiters = [
        "\nRéférences\n",
        "\nReferences\n",
        "\nNotes\n",
        "Apparat critique" ,
        "Aparat critique",    
        "External Links",
        "ALS",
        "ACS",
        ]
    for delimiter in delimiters:
        text = text.split(delimiter)[0]

    # delete footnote numbers "poincaré4" -> "poincaré", "poincaré.4" -> "poincaré"
    text = re.sub(
        r"(\w)(\d+)",
        r"\1 ",
        text
        )
    text = re.sub(
        r"(\w).(\d+)",
        r"\1 ",
        text
    )
    return text


def quick_clean_string(string:str) -> str:
    """quickly remove some unnecessary charactares from a string"""
    # the transcriptions come to the footnote numbers added to the end of the words in which they appear.
    # these removes those numbers
    result = string.lower()
    result = result.rstrip("0123456789").strip().rstrip("0123456789").strip()
    result = result.split("\(")[0]  #getting rid of LaTeX

    return result