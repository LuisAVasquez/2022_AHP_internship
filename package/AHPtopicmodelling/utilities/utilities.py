
from numpy import array
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


def get_clean_transcription(text):
    """delete the "Apparat critique", "Notes", and "References" sections"""
    delimiters = [
        "\nRéférences\n",
        "\nReferences\n",
        "\nNotes\n",
        "Apparat critique" ,
        "Aparat critique",    
        ]
    for delimiter in delimiters:
        text = text.split(delimiter)[0]

    return text


