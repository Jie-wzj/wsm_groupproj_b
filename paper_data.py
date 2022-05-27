# -*- coding: utf-8 -*-
import pandas as pd
from scipy import sparse

from copy import deepcopy
import struct
import nltk
import json
import tarfile
import io
import collections

from utils import *
from config import *

nltk.data.path.append(nltk_download_data_path)


class SortArg:
    base_order = {'score': 0, 'publish_time': 1, 'citations': 2}
    max_len = 3

    def __init__(self, order: list = None, ascending: list = None, mode=0):

        order = [i for i in range(self.max_len)] if order is None else order
        ascending = [False for _ in range(self.max_len)] if ascending is None else ascending

        self.order = np.array(order, dtype=np.int)
        self.ascending = ascending
        self.mode = mode

        self.arg_num = (self.order > -1).sum()

        if self.arg_num == 0:
            self.default()

    def default(self):
        self.order = [i for i in range(self.max_len)]
        self.ascending = [False for _ in range(self.max_len)]
        self.arg_num = self.max_len

    def pop(self, params):
        if not isinstance(params, list):
            params = [params]

        for param in params:
            if isinstance(param, str):
                if param in self.base_order:
                    self.order[self.base_order[param]] = -1
            elif isinstance(param, int) and -1 < param < self.max_len:
                self.order[param] = -1

        return self

    def arg(self):
        idx_order = np.argsort(self.order)
        base_order = list(self.base_order.keys())

        arg_by = []
        arg_ascending = []
        for idx in idx_order:
            if self.order[idx] > -1:
                arg_by.append(base_order[idx])
                arg_ascending.append(self.ascending[idx])

        return dict(by=arg_by, ascending=arg_ascending)

    def relevance_mode(self):
        return self.mode


class PaperData:
    def __init__(self, fltype=fltype, path: DataPath = data_path, config: Config = config, ):
        self.path = path
        self.fltype = fltype
        self.config = config

        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.stopwords = nltk.corpus.stopwords.words('english')

        print('\n============ Data init ============')
        # load md=metadata, doc_keys, numDoc
        if not os.path.isfile(self.path.metadata):
            self.count_citations().save_metadata()
        else:
            self.md = self.load_metadata()

        self.doc_keys = self.md.keys()
        self.numDoc = self.md.shape[0]

        # term_dict, numWord, numTerm, creat term_post, term_tf
        if not (os.path.isfile(self.path.term_dict) and os.path.isfile(self.path.postings) and os.path.isfile(
                self.path.tf)):
            self.process_base_data()
        else:
            self.load_dict()

        # calculation or load tf-idf, word_vector
        if not (os.path.isfile(self.path.word_vector) and os.path.isfile(self.path.tfidf)):
            self.calc_word_vector().reconstruct_tfidf().save_word_vector()
        else:
            self.load_word_vector()

        print('============ The data is ready! ============\n')

    def load_metadata(self, path=None):
        if path is None:
            path = self.path.metadata

        print('Load metadata from ', path)
        return pd.read_csv(path, low_memory=False)

    def count_citations(self):
        print('Count citations')
        self.md = self.load_metadata(self.path.metadata_raw)
        self.md['citations'] = 0
        self.md['score'] = 0
        for path in self.path.collections:
            tar = tarfile.open(path, "r:gz")
            for tarinfo in tar:
                if tarinfo.isreg():
                    f = tar.extractfile(tarinfo.name)
                    data = json.loads(f.read())
                    # self.md.query(f"sha == {data['paper_id']}").citations = len(data['bib_entries'])
                    self.md.loc[self.md.sha == data['paper_id'], 'citations'] = len(data['bib_entries'])
        return self

    def save_metadata(self):
        print('Save metadata to ', self.path.metadata)
        self.md.to_csv(self.path.metadata)

    def process_base_data(self):
        print('Metadata process')
        dictionary = dict()  # key: term, value: [postings list]

        # counter for the number of docs indexed
        for docID in range(self.numDoc):
            tokens = []
            if self.config.TITLE:
                doc = self.md.title[docID]
                if not pd.isna(doc):
                    tokens += nltk.word_tokenize(doc)
            if self.config.ABSTRACT:
                doc = self.md.abstract[docID]
                if not pd.isna(doc):
                    tokens += nltk.word_tokenize(doc)

            for token in tokens:
                term = token.lower()

                if self.config.IGNORE_STOPWORDS and term in self.stopwords:
                    continue  # if ignoring stopwords
                if self.config.IGNORE_NUMBERS and is_number(term):
                    continue  # if ignoring numbers

                term = self.stemmer.stem(term)  # stemming
                if term[-1] == "'":
                    term = term[:-1]  # remove apostrophe

                if self.config.IGNORE_SINGLES and len(term) == 1:
                    continue  # if ignoring single terms

                # if term not already in dictionary
                if term not in dictionary:
                    dictionary[term] = dict(dID=[docID], tf=[1], totf=1)  # define new term in in dictionary
                # else if term is already in dictionary
                else:
                    dictionary[term]['totf'] += 1
                    # if current docID is already in the postings list for term
                    if dictionary[term]['dID'][-1] == docID:
                        dictionary[term]['tf'][-1] += 1
                    # if current docID is not yet in the postings list for term, append it
                    else:
                        dictionary[term]['dID'].append(docID)
                        dictionary[term]['tf'].append(1)

        # save
        # open files for writing
        # dict_file = codecs.open(self.dictionary_file, 'w', encoding='utf-8')
        dict_file = open(self.path.term_dict, 'w', encoding='utf-8')
        post_file = open(self.path.postings, 'wb')
        tf_file = open(self.path.tf, 'wb')

        # build dictionary file and postings file
        byte_offset = 0  # byte offset for pointers to postings file
        total_f = 0

        for i, (term, term_dict) in enumerate(dictionary.items()):
            df = len(term_dict['dID'])

            # write each posting and tf into postings file
            for idx in range(df):
                # pack docID into a byte array of size 4
                post_file.write(struct.pack('I', term_dict['dID'][idx]))
                tf_file.write(struct.pack('I', term_dict['tf'][idx]))

            term_dict['tid'] = i
            term_dict['md'] = df
            term_dict['byte_offset'] = byte_offset
            del term_dict['dID'], term_dict['tf']

            byte_offset += self.config.BYTE_SIZE * df
            total_f += term_dict['totf']

        self.numTerm = len(dictionary)
        self.numWord = total_f
        self.term_dict = dictionary

        # write to dictionary file and update byte offset
        dictionary_n = dict(path=self.path.metadata,
                            numDoc=self.numDoc,
                            numTerm=self.numTerm,
                            numWord=self.numWord,
                            data=self.term_dict,
                            )

        print('Save term dictionary to ', self.path.term_dict)
        json.dump(dictionary_n, dict_file, ensure_ascii=False, indent=4, separators=(',', ':'))

        # close files
        dict_file.close()
        post_file.close()
        tf_file.close()

    def load_dict(self):
        print('Load term dictionary from ', self.path.term_dict)
        with open(self.path.term_dict, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)

        self.numTerm = dictionary['numTerm']
        self.numWord = dictionary['numWord']
        self.term_dict = dictionary['data']

        return self

    def load_posting_or_tf_list(self, offset, length, ispostings=True):
        posting_or_tf_list = []
        with io.open(self.path.postings if ispostings else self.path.tf, 'rb') as f:
            f.seek(offset)
            for i in range(length):
                posting_or_tf = f.read(self.config.BYTE_SIZE)
                posting_or_tf_list.append(struct.unpack('I', posting_or_tf)[0])
        return posting_or_tf_list

    def calc_word_vector(self):
        print('Calculation tf-idf, word vector')

        tf_vct = sparse.dok_matrix((self.numTerm, self.numDoc), dtype=self.fltype)
        idf_vct = np.zeros(self.numTerm, dtype=self.fltype)
        tfidf_vct = sparse.dok_matrix((self.numTerm, self.numDoc), dtype=self.fltype)

        posting_file = io.open(self.path.postings, 'rb')
        tf_file = io.open(self.path.tf, 'rb')

        for i, (term, term_dict) in enumerate(self.term_dict.items()):
            df = term_dict['md']
            idf = np.log10(self.numDoc / df)
            idf_vct[i] = idf
            for _ in range(df):
                posting = int(struct.unpack('I', posting_file.read(self.config.BYTE_SIZE))[0])
                tf_td = int(struct.unpack('I', tf_file.read(self.config.BYTE_SIZE))[0])
                tf = 1 + np.log10(tf_td)
                tf_vct[i, posting] = tf

                tfidf = tf * idf
                tfidf_vct[i, posting] = tfidf

        posting_file.close()
        tf_file.close()

        self.idf = idf_vct
        self.tfidf = tfidf_vct.copy().tocsc()

        tfidf_norm = sparse.linalg.norm(tfidf_vct, axis=0)
        for indx, v in tfidf_vct.items():
            tfidf_vct[indx] = v / tfidf_norm[indx[1]]

        self.word_vector = tfidf_vct.tocsc()

        return self

    def reconstruct_tfidf(self):
        if self.config.SVD_RECONSTRUCT:
            print('Reconstruct tfidf by svd')
            try:
                L, S, R = np.linalg.svd(self.tfidf.toarray())

                def top_n_pad_0(U, S, V, n):
                    t = list(S[:n])
                    for i in range(len(S) - n):
                        t.append(0)
                    A = np.diag(t)
                    newrow = [0] * len(S)
                    if len(U) > len(V):
                        for i in range(len(U) - len(S)):
                            A = np.vstack([A, newrow])
                        return A
                    else:
                        for i in range(len(V) - len(S)):
                            A = np.vstack([A.T, newrow]).T
                        return A

                def reconstruct(u, s, v, n):
                    A = top_n_pad_0(u, s, v, n)
                    return np.round((u.dot(A)).dot(v), decimals=3)

                def frobenius(a, a2):
                    a = np.array(a)

                    return (np.sqrt(np.sum((a - a2) ** 2))) / np.sqrt(np.sum(a ** 2))

                def find_k(num):
                    for i in range(1, num):
                        f = frobenius(self.tfidf, reconstruct(L, S, R, i))
                        # print(f)
                        if f < 0.38:
                            return i

                self.tfidf_r = reconstruct(L, S, R, find_k(len(S)))

                self.word_vector = self.tfidf_r / np.linalg.norm(self.tfidf_r, axis=0)

            except MemoryError:
                print('ERROR: Insufficient Memory')
            except:
                print('ERROR: Unknow Error')

        return self

    def save_word_vector(self):
        print('Save tf-idf to ', self.path.tfidf)
        sparse.save_npz(self.path.tfidf, self.tfidf, True)
        print('Save word vector to ', self.path.word_vector)
        sparse.save_npz(self.path.word_vector, self.word_vector, True)

    def load_word_vector(self):
        self.idf = np.zeros(self.numTerm, dtype=self.fltype)
        for i, (term, term_dict) in enumerate(self.term_dict.items()):
            self.idf[i] = math.log10(self.numDoc / term_dict['md'])

        print('Load tf-idf from ', self.path.tfidf)
        self.tfidf = sparse.load_npz(self.path.tfidf)
        print('Load word vector from ', self.path.word_vector)
        self.word_vector = sparse.load_npz(self.path.word_vector)

        return self

    def calc_query_v(self, mode):
        print('Calculation query vector')
        query_v = sparse.lil_matrix((1, self.numTerm), dtype=self.fltype)
        for token in self.tokens:
            if token in self.term_dict:
                query_v[0, self.term_dict[token]['tid']] += 1

        if mode == 0:
            query_v = query_v.T.tocsc()
        elif mode == 1:
            query_v = query_v.multiply(self.idf).T.tocsc()

        # self.query_v = query_v / np.linalg.norm(query_v)
        self.query_v = query_v / sparse.linalg.norm(query_v)

        return self

    def calc_score(self):
        # cosine
        print('Calculation Semantic Relevance score')
        self.md['score'] = self.query_v.multiply(self.word_vector).sum(axis=0).tolist()[0]

        # self.result_sort = sorted(zip(range(1, len(score) + 1), score), key=lambda x: x[1], reverse=True)
        return self

    def query(self, query):
        self.query_raw = query

        # prepare query list
        query = self.query_raw
        query = query.replace('(', '( ')
        query = query.replace(')', ' )')
        query = query.split(' ')

        self.tokens = collections.deque(shunting_yard(query))  # get query in postfix notation as a queue

        return self

    def search(self, query, arg: SortArg = None):
        if arg is None:
            arg = SortArg([], [])

        self.query(query).judge_boolean()

        if self.is_boolean:
            self.boolean_search()
        else:
            self.semantic_relevance(mode=arg.relevance_mode())

        return self.sort(arg=arg)

    def judge_boolean(self):
        self.is_boolean = False

        operators = ['NOT', 'OR', 'AND']
        for opt in operators:
            if opt in self.tokens:
                self.is_boolean = True
                break
        return self.is_boolean

    def semantic_relevance(self, mode=0):
        self.calc_query_v(mode=mode).calc_score()
        self.choose_list = self.md[self.md.score > 0].index

        return self

    def boolean_search(self):
        results_stack = []
        postfix_queue = deepcopy(self.tokens)
        while postfix_queue:
            token = postfix_queue.popleft()
            result = []  # the evaluated result at each stage
            # if operand, add postings list for term to results stack
            if token != 'AND' and token != 'OR' and token != 'NOT':
                token = self.stemmer.stem(token)  # stem the token
                # default empty list if not in dictionary
                if token in self.term_dict:
                    result = self.load_posting_or_tf_list(self.term_dict[token]['byte_offset'],
                                                          self.term_dict[token]['md'])

            # else if AND operator
            elif token == 'AND':
                right_operand = results_stack.pop()
                left_operand = results_stack.pop()
                # print(left_operand, 'AND', left_operand) # check
                result = boolean_AND(left_operand, right_operand)  # evaluate AND

            # else if OR operator
            elif token == 'OR':
                right_operand = results_stack.pop()
                left_operand = results_stack.pop()
                # print(left_operand, 'OR', left_operand) # check
                result = boolean_OR(left_operand, right_operand)  # evaluate OR

            # else if NOT operator
            elif token == 'NOT':
                right_operand = results_stack.pop()
                # print('NOT', right_operand) # check
                result = boolean_NOT(right_operand, list(range(self.numDoc)))  # evaluate NOT

            # push evaluated result back to stack
            results_stack.append(result)
            # print ('result', result) # check

        # NOTE: at this point results_stack should only have one item and it is the final result
        if len(results_stack) != 1:
            print("ERROR: results_stack. Please check valid query")  # check for errors

        self.choose_list = pd.Index(results_stack.pop())
        return self

    def sort(self, arg: SortArg = None):
        if arg is None:
            arg = SortArg([], [])
        if self.is_boolean:
            arg.pop('score')

        return self.md.loc[self.choose_list].sort_values(**arg.arg())


if __name__ == '__main__':
    mdata = PaperData()

    pdd = mdata.search('new clinic immunolog')

    print(pdd[['score', 'publish_time']])

    print()
