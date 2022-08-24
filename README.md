# 2022_AHP_internship



## Installation

1. Create and activate a virtual environment.

2. Install the required packages:
```sh
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

3. Install this package:
  - 3.1 Option 1: Clone the repository 
  ```sh
  git clone https://github.com/LuisAVasquez/2022_AHP_internship.git
  ```
  OR
  
  - 3.1 Option 2: 
      - Download the ZIP file (upper right of [AHP NLP tools](https://github.com/LuisAVasquez/2022_AHP_internship), `Code` > `Download ZIP`)  
      - decompress it in the main folder
      - delete the branch name (`-<branch_name>`, for example, `-master`) that is at the end of the name of the decompressed folder. 
  
  - 3.2: Install locally
  ```sh
  pip install -e 2022_AHP_internship/.
  ```

## Usage

- There is a tool available for cleaning the text of the letters of the Henri Poncaré correspondence (deleteting superfluous characters, ignoring the critical section and the references)

```python
from AHPNLP.utilities.utilities import get_clean_transcription
get_clean_transcription("<raw transcription as retrieved from the collection>")
```

gives:
```python
<clean transcription>
```

- There is a (french) tokenizer to extract normalized content tokens:

```python
clean_letter_transcription = """
"Paris le  août  \nÉcole Polytechnique\nMonsieur H. Poincarré, ing. des Ponts et chaussées, professeur à la faculté de Caen.\nMon cher camarade,\nJ'ai vérifié que la  partie de votre mémoire, la partie algébrique s'arrêtant à la page   sur les   pages dont il se compose, formerait, en effet, les  ou  feuilles dont nous disposons dans le   cahier qui est sous presse. Je préviens M. Gauthier-Villars de cette solution; vous aurez seulement à fournir un second titre, à annoncer la suite de votre mémoire à la fin du chapitre V et à modifier légèrement la rédaction du commencement du chapitre VI \nRecevez, mon cher camarade, l'assurance de mes sentiments affectueux.\nLe Directeur des Études.\nA. Laussedat\n\n\xa0\n\n"
"""
from AHPNLP.tokenization.tokenization import Tokenizer
standard_tokenizer = Tokenizer()
content_tokens = standard_tokenizer(clean_letter_transcription)
content_tokens
```
gives

```python
>>> ['paris', 'août', 'école', 'polytechnique', 'poincarré', 'pont', 'chausser', 'professeur', 'faculté', 'caen', 'cher', 'camarade', 'vérifier', 'partie', 'mémoire', 'partie', 'algébrique', 'arrêter', 'page', 'page',
'compose', 'former', 'feuille', 'disposer', 'cahier', 'presse', 'prévenir', 'gauthier', 'villars', 'solution',
'fournir', 'second', 'titre', 'annoncer', 'suite', 'mémoire', 'chapitre', 'modifier', 'légèrement', 'rédaction',
'commencement', 'chapitre', 'recevez', 'cher', 'camarade', 'assurance', 'sentiment', 'affectueux', 'directeur',
 'étude', 'laussedat']
```

- There is also functionality for extracting nominal groups (noun + some complement)

```python
nominal_groups_tokenizer = Tokenizer(nominal_groups=True)
nominal_groups = nominal_groups_tokenizer(clean_letter_transcription)
nominal_groups
```

gives

```python
['partie algébrique', 'école polytechnique', 'chapitre v', 'sentiments affectueux', 'directeur des études',
'faculté de caen']
```


## Triples

Triples for content tokens and nominal groups were added to the SPARQL endpoint of the Henri Poincaré archives. The folder `triples` contains notebooks with instructions to replicate the creation of these triples.