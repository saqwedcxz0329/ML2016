import pandas as pd
import numpy as np

dtypes = {'uuid': np.str_, 'document_id': np.int32, 'timestamp': np.int32, 'platform': np.int32, 'geo_location': np.str_}

train = pd.read_csv("./data/clicks_train.csv")
promoted = pd.read_csv("./data/promoted_content.csv")
docCategories = pd.read_csv("./data/documents_categories.csv")
idx = docCategories.groupby(["document_id"])["confidence_level"].transform(max) == docCategories['confidence_level']
docCategories = docCategories[idx]
docTopics = pd.read_csv("./data/documents_topics.csv")
idx = docTopics.groupby(["document_id"])["confidence_level"].transform(max) == docTopics['confidence_level']
docTopics = docTopics[idx]

#print promoteda
#print promoted.loc[promoted["ad_id"] == 1]
#print docCategories.loc[docCategories["document_id"] == 691188]

print docTopics.loc[ docTopics["document_id"]== 514201]


for adID in train["ad_id"]:
    adDocID = promoted.loc[promoted["ad_id"] == adID]
    docID = adDocID["document_id"].iloc[0]
    #print docIdIndex
    dframeCategories =  docCategories.loc[docCategories["document_id"] == docID]
    #dframeCategories = dframeCategories.loc[dframeCategories["confidence_level"] > 0.5]a
    
    dframeTopics =  docTopics.loc[docTopics["document_id"] == docID]
    #dframeTopics = dframeTopics.loc[dframeTopics["confidence_level"] > 0.5]
    
    print "ad id: %d" %adID
    print "document id: %d" %docID
    print dframeCategories["category_id"].iloc[0]
    print dframeTopics["topic_id"].iloc[0]


#ad_likelihood = train.groupby('document_id').clicked.agg(['count','sum','mean']).reset_index()
#M = train.clicked.mean()
del train
