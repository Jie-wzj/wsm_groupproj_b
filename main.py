# -*- coding: utf-8 -*-


from flask import Flask, render_template, request

import sqlite3
import configparser
import time
import boolean_search
import pandas as pd

import jieba

app = Flask(__name__)

db_path = './data/metadata.db'
global page
global keys


@app.route('/')
def main():
    global df
    df = pd.read_csv('./data/metadata.csv')
    print('load complete')
    return render_template('search.html', error=True)


# 读取表单数据，获得doc_ID
@app.route('/search/', methods=['POST'])
def search():
    global keys
    global checked
    checked = ['checked="true"', '', '']
    keys = request.form['key_word']
    print(keys)
    if keys not in ['']:
        # print(time.clock())
        docs = find(keys)
        # print(time.clock())
        return render_template('high_search.html', checked=checked, key=keys, docs=docs, error=True)
    else:
        return render_template('search.html', error=False)


# 将需要的数据以字典形式打包传递给search函数
def find(search_keys, extra=False):
    global db_path
    f = open('query', 'w')
    f.write(search_keys)
    f.close()
    boolean_search.search('dict', 'post', 'query', 'out')
    f = open('out', 'r')
    docs = []
    doc_list = f.read()
    f.close()
    doc_list = doc_list.split(' ')
    print(doc_list)
    for i in range(len(doc_list)):
        doc_index = int(doc_list[i])
        tmp = df.iloc[doc_index]
        tmp = tmp.to_dict()
        docs.append(tmp)

    # docs =get_k_nearest(db_path=db_path,search_keys=search_keys)
    return docs


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def get_k_nearest(db_path, search_keys, k=50):
    conn = sqlite3.connect(db_path)
    conn.row_factory = dict_factory
    c = conn.cursor()
    c.execute("SELECT * FROM metadata WHERE source_x =?", (search_keys,))
    docs = c.fetchall()
    # print(docs)
    conn.close()
    return docs[0:k]  # max = k


if __name__ == '__main__':
    jieba.initialize()  # 手动初始化（可选）
    app.run()
