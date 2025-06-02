import numpy as np
from typing import Dict

class Scorer:
  def __init__(
    self,
    lexicon: Dict,
    document_index: Dict,
    total_documents: int,
    average_document_token_count: float,
    k1: float = 1.5,
    b: float = 0.75
  ):
    """
    Initialize the Scorer with corpus statistics.

    Args:
      lexicon (Dict): Mapping from token to the document frequency and corpus frequency.
      document_index (Dict): Mapping from docid to character count and token count.
      total_documents (int): Total number of documents in the corpus.
      average_document_token_count (float): Average document token count in the corpus.
      k1 (float): BM25 term frequency saturation parameter.
      b (float): BM25 length normalization parameter.
    """
    self.lexicon = lexicon
    self.document_index = document_index
    self.total_documents = total_documents
    self.average_document_token_count = average_document_token_count
    self.k1 = k1
    self.b = b
    
    # Cache for IDF values to avoid recomputation
    self._idf_cache: Dict[str, float] = {}

  def compute_idf(self, token: str) -> float:
    """
    Compute the Inverse Document Frequency (IDF) for a token using BM25 formula.

    Args:
      token (str): Token for which to compute IDF.

    Returns:
      float: IDF score.
    """
    if token in self._idf_cache:
      return self._idf_cache[token]

    token_info = self.lexicon.get(token)
    if not token_info:
      return 0.0

    df = token_info['document_frequency']
    idf = np.log(1 + (self.total_documents - df + 0.5) / (df + 0.5))

    self._idf_cache[token] = idf
    return idf

  def compute_tfidf(self, token: str, term_frequency: int, docid: str) -> float:
    """
    Compute TF-IDF score for a token in a document.

    Args:
      token (str): Token to score.
      term_frequency (int): Frequency of token in the document.
      docid (str): Document ID.

    Returns:
      float: TF-IDF score.
    """
    token_info = self.lexicon.get(token)
    doc_info = self.document_index.get(docid)
    if not token_info or not doc_info:
      return 0.0

    tf = term_frequency / doc_info['token_count']
    df = token_info['document_frequency']
    idf = np.log((self.total_documents + 1) / (df + 1))

    return tf * idf

  def compute_bm25(self, token: str, term_frequency: int, docid: str) -> float:
    """
    Compute BM25 score for a token in a document.

    Args:
      token (str): Token to score.
      term_frequency (int): Frequency of token in the document.
      docid (str): Document ID.

    Returns:
      float: BM25 score.
    """
    idf = self.compute_idf(token)

    doc_info = self.document_index.get(docid)
    token_count = doc_info['token_count']

    numerator = term_frequency * (self.k1 + 1)
    denominator = term_frequency + self.k1 * (
      1 - self.b + self.b * (token_count / self.average_document_token_count)
    )

    if denominator == 0:
      return 0.0

    return idf * (numerator / denominator)
