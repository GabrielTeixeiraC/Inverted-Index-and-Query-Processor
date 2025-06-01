import numpy as np

class Scorer:
    def __init__(self, lexicon, document_index, total_docs=0, avg_doc_length=0, k1=1.5, b=0.75):
        """
        Initialize the Scorer with a lexicon.
        Args:
            lexicon (dict): Lexicon containing document frequencies for terms.
            total_docs (int): Total number of documents in the corpus.
            avg_doc_length (float): Average document length in the corpus.
            k1 (float): Term frequency saturation parameter.
            b (float): Length normalization parameter.
        """
        self.lexicon = lexicon
        self.document_index = document_index
        self.total_docs = total_docs
        self.avg_doc_length = avg_doc_length
        self.k1 = k1
        self.b = b
        self.idf_cache = {}

    def idf(self, token):
        """
        Compute the IDF for a token, with caching.
        """
        if token in self.idf_cache:
            return self.idf_cache[token]

        token_info = self.lexicon.get(token)

        df = token_info['doc_frequency']
        idf = np.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))

        self.idf_cache[token] = idf
        return idf

    def tfidf(self, token, term_frequency: int):
        """
        Compute TF-IDF score.
        """
        token_info = self.lexicon.get(token)
        if not token_info:
            return 0.0

        document_frequency = token_info['doc_frequency']
        idf = np.log((self.total_docs + 1) / (document_frequency + 1))
        return term_frequency * idf

    def bm25(self, token, term_frequency: int, docid: str):
        idf = self.idf(token)
        doc_length = self.document_index[str(docid)]['doc_length']

        numerator = term_frequency * (self.k1 + 1)
        denominator = term_frequency + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

        return idf * (numerator / denominator) if denominator != 0 else 0.0


if __name__ == "__main__":
    # Example usage
    scorer = Scorer('./tmp/term_lexicon.jsonl')
    print(scorer.lexicon['appl'])  # This will print the loaded lexicon
    # You can add more functionality to test the scoring methods here.