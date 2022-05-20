import os
import shutil
import json
from datetime import datetime
import sqlite3
import re

import pandas as pd
from tqdm import tqdm


root_path = os.path.join(os.path.split(__file__)[0], './data')
root_path = os.path.abspath(root_path)


conn = sqlite3.connect(os.path.join(root_path, 'metadata.db'))
conn.text_factory = str
print("Opened database successfully...")
c = conn.cursor()

def deal_data(filepath):
    '''
        Deal with data files.
        :param filepath: path name
        :return: None
    '''



    # '''Create table metadata'''
    # sql = '''CREATE TABLE metadata
    #             (cord_uid        TEXT   ,
    #             sha            TEXT  ,
    #             source_x         TEXT,
    #             title       TEXT,
    #             doi           TEXT,
    #             pmcid          TEXT,
    #             pubmed_id        TEXT,
    #             license     TEXT,
    #             abstract         TEXT,
    #             publish_time   TEXT,
    #             authors        TEXT,
    #             journal       TEXT,
    #             Microsoft Academic Paper ID        FLOAT,
    #             WHO_Covidence        TEXT,
    #             has_pdf_parse       TEXT,
    #             has_pmc_xml_parse       TEXT,
    #             full_text_file       TEXT,
    #             url       TEXT);'''
    #
    # c.execute(sql)
    # print("Table created successfully...")
    # conn.commit()

    keys = ['cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id',
       'license', 'abstract', 'publish_time', 'authors', 'journal',
       'Microsoft Academic Paper ID', 'WHO #Covidence', 'has_pdf_parse',
       'has_pmc_xml_parse', 'full_text_file', 'url']


    '''Save data to the database'''
    with open(filepath , 'r', encoding='utf-8') as f:
        pd_data = pd.read_csv(f, encoding="utf-8")

        # data processing
        for idx in pd_data.index:
            values = []
            for key in keys:
                values.append(pd_data.loc[idx][key])
            values = tuple(values)
            sql = "insert into metadata values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            c.execute(sql, values)

    conn.commit()
    print("Records created successfully...")
    conn.close()

deal_data('./data/metadata.csv')