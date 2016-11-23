from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

categories = ['wordpress', 'oracle', 'svn', 'apache', 'excel', 'matlab', 'visual-studio',
'cocoa', 'osx', 'bash', 'spring', 'hibernate', 'scala', 'sharepoint', 'ajax', 'qt', 'drupal',
'linq', 'haskell', 'magento']

cluster_num = 20

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
    del check_index
    return check_list

def predict(predict_title, check_list):
    predict_file = open("predict.csv", "w")
    predict_file.write("ID,Ans\n")
    for i in range(len(check_list)):
        row = check_list[i]
        x_index = row['x_ID']
        y_index = row['y_ID']
        ans = int(predict_title[x_index] == predict_title[y_index])
        predict_file.write("%d,%d\n" %(i, ans))

doc = open('docs.txt', "r")
title = open('title_StackOverflow.txt', 'r')
check_list = parseData()

vectorizer = TfidfVectorizer(max_df=0.5, max_features=500,
                                 min_df=2, stop_words='english')
# Generate word vector
vectorizer.fit(doc)

# Transform title to word vector
X = vectorizer.transform(title)


svd = TruncatedSVD(2)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)


km = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1,
                verbose=True)

# Fit and predict the title
predict_title = km.fit_predict(X)
predict(predict_title, check_list)

for i in range(20):
    point = X[np.where(predict_title == i)]
    point = np.transpose(point)
    plt.plot(point[0], point[1], '.')

plt.show()
