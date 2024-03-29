import pandas as pd
import numpy as np

def groupbyConfidence(dframe):
    # Only leave the higher confidence value
    idx = dframe.groupby(["document_id"])["confidence_level"].transform(max) == dframe['confidence_level']
    return dframe[idx]

def getID(row, col, dframe):
    # Get the doucment_id through display_id(event.csv) or ad_id(promoted_content.csv)
    elementID = row[col]
    elementDocID = dframe.loc[dframe[col] == elementID]
    ID = elementDocID["document_id"].iloc[0]
    return ID

def getFeature(dframe, docID, col):
    # Get the feature through documet_*.csv file
    # docID: document_id
    dframe = dframe.loc[dframe["document_id"] == docID]
    if dframe.empty:
        return 0
    else:
        num = dframe[col].iloc[0]
        return num

train = pd.read_csv("./data/clicks_train.csv")
promoted = pd.read_csv("./data/promoted_content.csv")
events = pd.read_csv("./data/events.csv")
docCategories = pd.read_csv("./data/documents_categories.csv")
docCategories = groupbyConfidence(docCategories)
docTopics = pd.read_csv("./data/documents_topics.csv")
docTopics = groupbyConfidence(docTopics)
docMeta = pd.read_csv("./data/documents_meta.csv")
features_file = open("features.txt", "w")

print "Start..."
adID = train["ad_id"].tolist()
promoted = promoted[promoted["ad_id"].isin(adID)]
#promoted.to_csv("./process_promoted.csv")
docID = promoted["document_id"].tolist()
docCategoriesForAd = docCategories[docCategories["document_id"].isin(docID)]
docTopicsForAd = docTopics[docTopics["document_id"].isin(docID)]

displayID = train["display_id"].tolist()
events = events[events["display_id"].isin(displayID)]
docID = events["document_id"].tolist()
docCategoriesForEvent = docCategories[docCategories["document_id"].isin(docID)]
docTopicsForEvent = docTopics[docTopics["document_id"].isin(docID)]

del docCategories
del docTopics

for index, row in train.iterrows():
    print index
    label = row["clicked"]

    display_id = row["display_id"]
    docID = events.iloc[[display_id-1]]["document_id"].iloc[0]
    #docID = getID(row, "display_id", events)
    category = getFeature(docCategoriesForEvent, docID, "category_id")
    topic = getFeature(docTopicsForEvent, docID, "topic_id")
    source = getFeature(docMeta, docID, "source_id")
    
    #print "By display_id... %d" %docID
    #print "Category: %d" %category
    #print "Topic: %d" %topic

    features_file.write("%s %s %s " %(str(category), str(topic), str(source)))
    adID = getID(row, "ad_id", promoted)
    category = getFeature(docCategoriesForAd, adID, "category_id")
    topic = getFeature(docTopicsForAd, adID, "topic_id")
    source = getFeature(docMeta, adID, "source_id")
    
    #print "By ad_id... %d" %adID
    #print "Category: %d" %category
    #print "Topic: %d" %topic
    #print "source: %d" %source
    features_file.write("%s %s %s %s \n" %(str(category), str(topic), str(source), str(label)))

#ad_likelihood = train.groupby('document_id').clicked.agg(['count','sum','mean']).reset_index()
#M = train.clicked.mean()

del train
