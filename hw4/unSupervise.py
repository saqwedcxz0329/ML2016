from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from gensim.models import doc2vec
from stop_words import get_stop_words
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import re
import os

cluster_num = 20
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
data_directory = sys.argv[1]
output_file = sys.argv[2]
title_path = data_directory +  '/title_StackOverflow.txt'
check_index_path = data_directory + '/check_index.csv'
doc_path = data_directory + '/docs.txt'
trueLabel_path = data_directory + '/label_StackOverflow.txt'

def parseData():
    check_index = open(check_index_path, 'r')
    check_list = []
    check_index.readline()
    for line in check_index.readlines():
        row_data = {'x_ID': 0, 'y_ID': 0}
        line = line.split(',')
        row_data['x_ID'] = int(line[1])
        row_data['y_ID'] = int(line[2].split('\n')[0])
        check_list.append(row_data)
    check_index.close()
    del check_index
    return check_list

def parseTrueLabel():
    file = open(trueLabel_path, 'r')
    true_label = []
    for line in file.readlines():
        tmp = line.split('\n')[0]
        true_label.append(int(tmp))
    file.close()
    true_label = np.array(true_label, dtype='uint8')
    return true_label


def convertTitle(file):
    X = []
    for line in file:
        tmp = line.split()
        X.append(model.infer_vector(tmp))
    return X

def predict(predict_title, check_list):
    predict_file = open(output_file, "w")
    predict_file.write("ID,Ans\n")
    for i in range(len(check_list)):
        row = check_list[i]
        x_index = row['x_ID']
        y_index = row['y_ID']
        ans = int(predict_title[x_index] == predict_title[y_index])
        predict_file.write("%d,%d\n" %(i, ans))
    predict_file.close()

def preProcessingDoc(doc, output_file_name):
    file = open(output_file_name, "w+")
    stopwords = set(get_stop_words('english'))
    stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')',
         '[', ']', '{', '}', ' ', '==', '!=', '>', '<', '//', '};',
          '=', '--', '||', '&&', '+', '-', '*', '/', '_', '&', '#'])
    for line in doc:
        if line != '\n':
            text = ' '.join([word for word in line.lower().split() if word not in stopwords])
            if text != '':
                file.write(text)
                file.write('\n')
    doc.close()
    file.close()

def wordVector(doc):
    sentences=doc2vec.TaggedLineDocument(doc)
    model = doc2vec.Doc2Vec(size = 100, window = 300, min_count = 5, workers=10)
    model.build_vocab(sentences)
    model.train(sentences)

    return model

def dimensionReduction(X):
    svd = TruncatedSVD(20)
    normalizer = Normalizer(copy=True)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    return X

def visulization(X, predict_cluster, figure_num = 1):
    fig = plt.figure(figure_num)
    for i in range(cluster_num):
        point = X[np.where(predict_cluster == i)]
        point = np.transpose(point)
        plt.plot(point[0], point[1], 'o')
    plt.xlabel('X')
    plt.ylabel('Y')
    if figure_num == 1:
        plt.title('Predict Label Visulization')
        fig.canvas.set_window_title('Predict Label Visulization')
    else:
        plt.title('True Label Visulization')
        fig.canvas.set_window_title('True Label Visulization')
    fig.show()

def findMostCommonWords(title_list, predict_cluster):
    print len(title_list)
    title_list = preprocess(title_list)
    for m in range(cluster_num):
        print "==============%s==============" %str(m)
        cluster_list = []
        file_name = "cluster_%s.txt" %str(m)
        file = open(file_name, 'w')
        for j in np.where(predict_cluster == m)[0]:
            cluster_list.append(title_list[j])
            file.write(title_list[j])
        file.close()

        cluster_file = open(file_name, 'r')
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=None,
                                 min_df=2, stop_words='english')
        X = vectorizer.fit_transform(cluster_list)
        print X.shape
        cluster_file.close()
        idf = vectorizer.idf_
        max_idf = max(idf)
        feature_names = vectorizer.get_feature_names()
        max_index = [i for i, j in enumerate(idf) if j == max_idf]
        good_words = open("goodWord_%s.txt" %str(m), 'w')
        for i in max_index:
            good_words.write(feature_names[i])
            good_words.write('\n')

def preprocess(data):
    for i in range(len(data)):
        parts = re.compile('\w+').findall(data[i])
        data[i]  = ' '.join([s.lower() for s in parts])
    return data

def getTitleList(file):
    title_list = []
    for line in file.readlines():
        title_list.append(line)
    file.close()
    return title_list

## Word vector ##
"""
preProcessingDoc(open(doc_path, "r"), 'process_doc.txt')
doc = open('process_doc.txt', 'r')
preProcessingDoc(open(title_path, 'r'), 'process_title.txt')
title_file = open('process_title.txt', 'r')

model = wordVector(doc)
X = convertTitle(title_file)

doc.close()
title_file.close()

km = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=500, n_init=1,
                verbose=True)

# Fit and predict the title
predict_title = km.fit_predict(X)
check_list = parseData()
predict(predict_title, check_list)
"""

###################################################################

## Tf-Idf ##

preProcessingDoc(open(title_path, 'r'), 'process_title.txt')


#print title_list
title_file = open('process_title.txt', 'r')

vectorizer = TfidfVectorizer(max_df=0.5, max_features=None,
                                 min_df=2, stop_words='english')

# Fit and transform title to features vector
X = vectorizer.fit_transform(title_file)

title_file.close()


X = dimensionReduction(X)


km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)

# Fit and predict the title
predict_cluster = km.fit_predict(X)
#check_list = parseData()
#predict(predict_cluster, check_list)

# Visualization
pca=PCA(n_components=2)
pca.fit_transform(X)
print "(data_number, dimension) " + str(X.shape)
visulization(X, predict_cluster,1)
true_cluster = parseTrueLabel()
visulization(X, true_cluster,2)

# Find good words
title_file = open('process_title.txt', 'r')
title_list = getTitleList(title_file)
title_file.close()
findMostCommonWords(title_list, predict_cluster)

raw_input()