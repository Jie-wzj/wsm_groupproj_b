import numpy as np
import os

COLLECTIONS = [
    'biorxiv_medrxiv.tar.gz',
    'comm_use_subset.tar.gz',
    'custom_license.tar.gz',
    'noncomm_use_subset.tar.gz',
]

nltk_download_data_path = 'D:\\programData\\datas_python\\nltk_data'


class Config:
    def __init__(self,
                 LIMIT=None,  # (for testing) to limit the number of documents indexed
                 IGNORE_STOPWORDS=True,  # toggling the option for ignoring stopwords
                 IGNORE_NUMBERS=True,  # toggling the option for ignoring numbers
                 IGNORE_SINGLES=True,  # toggling the option for ignoring single character tokens
                 RECORD_TIME=False,  # toggling for recording the time taken for indexer
                 BYTE_SIZE=4,  # docID is in int
                 TITLE=True,
                 ABSTRACT=False,

                 SVD_RECONSTRUCT=False,
                 ):
        self.LIMIT = LIMIT
        self.IGNORE_STOPWORDS = IGNORE_STOPWORDS
        self.IGNORE_NUMBERS = IGNORE_NUMBERS
        self.IGNORE_SINGLES = IGNORE_SINGLES
        self.RECORD_TIME = RECORD_TIME
        self.BYTE_SIZE = BYTE_SIZE

        self.TITLE = TITLE
        self.ABSTRACT = ABSTRACT

        self.SVD_RECONSTRUCT = SVD_RECONSTRUCT


class DataPath:
    def __init__(self,
                 base_path='data/',

                 metadata_raw='metadata.csv',
                 # collections=COLLECTIONS,

                 metadata='metadata_with_citations.csv',
                 term_dict='term.json',
                 postings='term_post',
                 tf='term_df',
                 tfidf='tfidf_n.npz',
                 word_vector='word_vector.npz',
                 ):
        self.base_path = base_path

        self.metadata_raw = os.path.join(base_path, metadata_raw)
        self.collections = [os.path.join(base_path, collection) for collection in COLLECTIONS]
        self.metadata = os.path.join(base_path, metadata)

        self.term_dict = os.path.join(base_path, term_dict)
        self.postings = os.path.join(base_path, postings)
        self.tf = os.path.join(base_path, tf)
        self.tfidf = os.path.join(base_path, tfidf)
        self.word_vector = os.path.join(base_path, word_vector)


data_path = DataPath(
    base_path='data/',

    metadata_raw='metadata.csv',

    metadata='metadata_with_citations.csv',
    term_dict='term.json',
    postings='term_post',
    tf='term_df',
    tfidf='tfidf_n.npz',
    word_vector='word_vector.npz',
)

config = Config(
    LIMIT=None,  # (for testing) to limit the number of documents indexed
    IGNORE_STOPWORDS=True,  # toggling the option for ignoring stopwords
    IGNORE_NUMBERS=True,  # toggling the option for ignoring numbers
    IGNORE_SINGLES=True,  # toggling the option for ignoring single character tokens
    RECORD_TIME=False,  # toggling for recording the time taken for indexer
    BYTE_SIZE=4,  # docID is in int
    TITLE=True,
    ABSTRACT=False,

    SVD_RECONSTRUCT=False,
)

fltype = np.float32
