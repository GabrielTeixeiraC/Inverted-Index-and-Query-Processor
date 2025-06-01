import json
import numpy as np

class Scorer:
    def __init__(self, lexicon, total_docs=0):
        """
        Initialize the Scorer with a lexicon path.
        Args:
            lexicon_path (str): Path to the lexicon file.
        """
        self.lexicon = lexicon 
        self.total_docs = total_docs
        self.idf_cache = {}

    def tfidf(self, token, term_frequency: int):
        """
        Calculate the TF-IDF score for a given token in a document.
        Args:
            token (str): The token for which to calculate the TF-IDF score.
            term_frequency (int): The frequency of the token in the document.
        Returns:
            float: The TF-IDF score for the token in the document.
        """
        token_info = self.lexicon.get(token)
        document_frequency = token_info['doc_frequency']
        idf = np.log((self.total_docs + 1) / (document_frequency + 1))
        return term_frequency * idf


if __name__ == "__main__":
    # Example usage
    scorer = Scorer('./tmp/term_lexicon.jsonl')
    print(scorer.lexicon['appl'])  # This will print the loaded lexicon
    # You can add more functionality to test the scoring methods here.