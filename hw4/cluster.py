from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import doc2vec
from stop_words import get_stop_words
import logging

cluster_num = 20
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def parseData():
    check_index = open('check_index.csv', 'r')
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

def parseTitle():
    file = open('process_title.txt', 'r')
    title_list = []
    for line in file:
        title_list.append(line.split())
    file.close()
    return title_list

def predict(predict_title, check_list):
    predict_file = open("predict.csv", "w")
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
    stopwords = set(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')',
         '[', ']', '{', '}', ' ', '==', '!=', '>', '<', '//', '};',
          '=', '--', '||', '&&', '+', '-', '*', '/', '_', '&', '#'])
    """
    stopwords = set(get_stop_words('english'))
    stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')',
         '[', ']', '{', '}', ' ', '==', '!=', '>', '<', '//', '};',
          '=', '--', '||', '&&', '+', '-', '*', '/', '_', '&', '#'])
    """
    for line in doc:
        if line != '\n':
            text = ' '.join([word for word in line.lower().split() if word not in stopwords])
            if text != '':
                file.write(text)
                file.write('\n')
    file.close()

def wordVector(doc, title):
    sentences=doc2vec.TaggedLineDocument(doc)
    model = doc2vec.Doc2Vec(size = 100, window = 300, min_count = 5, workers=10, alpha=0.025, min_alpha = 0.025)
    model.build_vocab(sentences)
    model.train(sentences)

    return model

def dimensionReduction(X):
    svd = TruncatedSVD(15)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    return X

def visulization(X):
    for i in range(20):
        point = X[np.where(predict_title == i)]
        point = np.transpose(point)
        plt.plot(point[0], point[1], '.')
    plt.show()

#preProcessingDoc(open('docs.txt', "r"), 'process_doc.txt')
#doc = open('process_doc.txt', 'r')
preProcessingDoc(open('title_StackOverflow.txt', 'r'), 'process_title.txt')
title = open('process_title.txt', 'r')

"""
model = wordVector(doc, title)
title_list = parseTitle()
for i in range(len(title_list)):
    title = title_list[i]
    title_list[i] = model.infer_vector(title)

km = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=500, n_init=1,
                verbose=True)

# Fit and predict the title
predict_title = km.fit_predict(title_list)
check_list = parseData()
predict(predict_title, check_list)
"""

###################################################################

vectorizer = TfidfVectorizer(max_df=0.5, max_features=None,
                                 min_df=2, stop_words='english')

# Generate features vector
#vectorizer.fit(doc)

# Transform title to features vector
#X = vectorizer.transform(title)

X = vectorizer.fit_transform(title)

print X.shape

X = dimensionReduction(X)

km = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1,
                verbose=True)

# Fit and predict the title
predict_title = km.fit_predict(X)
check_list = parseData()
predict(predict_title, check_list)

# Visualization
#visulization(X)
