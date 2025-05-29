import collections
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer 
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class Tokenizer:
  """
  A class to handle tokenization of text using NLTK.
  """
  
  def __init__(self):
    """
    Initializes the Tokenizer class.
    """
    # Ensure that the NLTK tokenizer is downloaded
    
    # Initialize the stemmer
    self.stemmer = SnowballStemmer('english')
    
    # Initialize the stop words
    self.stop_words = set(stopwords.words('english'))

  def tokenize(self, text: str) -> List[str]:
    """
    Tokenizes the input text into words using NLTK's word_tokenize function.
    
    Args:
        text (str): The input text to tokenize.
        
    Returns:
        list: A list of tokens (words).
    """
    if not isinstance(text, str):
      raise ValueError("Input text must be a string.")

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove punctuation, convert to lowercase and stem the words
    tokens = [self.stemmer.stem(word.lower()) for word in tokens if word.isalnum() and word not in self.stop_words and len(word) > 2]

    return tokens

if __name__ == "__main__":
  tokenizer = Tokenizer()

  text = "This is a test sentence, testing the tokenizer in a test case."
  tokens = tokenizer.tokenize(text)
  print(tokens)