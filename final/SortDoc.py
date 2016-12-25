import pandas as pd
import numpy as np
import sys

def groupbyConfidence(dframe):
    # Only leave the higher confidence value
    idx = dframe.groupby(["document_id"])["confidence_level"].transform(max) == dframe['confidence_level']
    return dframe[idx]

inputFolder = sys.argv[1]
outputFolder = sys.argv[2]

promoted = pd.read_csv("%s/promoted_content.csv" %inputFolder)
events = pd.read_csv("%s/events.csv" %inputFolder)
docCategories = pd.read_csv("%s/documents_categories.csv" %inputFolder)
docCategories = groupbyConfidence(docCategories)
docTopics = pd.read_csv("%s/documents_topics.csv" %inputFolder)
docTopics = groupbyConfidence(docTopics)
docMeta = pd.read_csv("%s/documents_meta.csv" %inputFolder)

print "Start..."

events.sort_values(by=["display_id"], ascending=[False]).to_csv("%s/order_events.csv" %outputFolder, index=False)
promoted.sort_values(by=["ad_id"], ascending=[False]).to_csv("%s/order_promoted_content.csv" %outputFolder, index=False)
docCategories.sort_values(by=["document_id"], ascending=[False]).to_csv("%s/order_documents_categories.csv" %outputFolder, index=False)
docTopics.sort_values(by=["document_id"], ascending=[False]).to_csv("%s/order_documents_topics.csv" %outputFolder, index=False)
docMeta.sort_values(by=["document_id"], ascending=[False]).to_csv("%s/order_documents_meta.csv" %outputFolder, index=False)

