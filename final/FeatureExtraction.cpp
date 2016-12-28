#include<iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <sstream>

using namespace std;

vector<string> split(string str, char symbol)
{
    string col;
    vector<string> rowVector;
    istringstream rowStream (str);
    while(getline(rowStream, col, symbol))
    {
        if(col=="")
        {
            col = "0";
        }
        rowVector.push_back(col);
    }
    return rowVector;
}

vector<vector<string> > convertToVector(string filePath)
{
    ifstream file;
    file.open(filePath);
    string row;
    vector<vector<string> > fileVector;
    getline(file, row);
    bool initFlag = false;
    while(getline(file, row))
    {
        vector<string> rowVector = split(row, ',');
        stringstream tmp(rowVector[0]);
        int dataIndex;
        tmp>> dataIndex;
        if(!initFlag)
        {
            dataIndex += 1;
            cout<<dataIndex<<endl;
            fileVector.resize(dataIndex);
            vector<string> initValue;
            for(int i = 0; i < rowVector.size(); i++)
            {
                initValue.push_back("0");
            }
            for(int i = 0; i < fileVector.size(); i++)
            {
                fileVector[i] = initValue;
            }
            initFlag = true;
        }
        fileVector[dataIndex] = rowVector;
    }
    file.close();
    return fileVector;
}

int main(int argc, char *argv[])
{
    ifstream trainDataFile;
    ofstream featuresFile;
    string inputFolder(argv[1]);
    string outputFile(argv[2]);
    trainDataFile.open(inputFolder + "/clicks_test.csv");
    featuresFile.open(outputFile);
    string promotedPath = inputFolder + "/order_promoted_content.csv";
    string eventPath = inputFolder + "/order_events.csv";
    string categoriesPath = inputFolder + "/order_documents_categories.csv";
    string topicsPath = inputFolder + "/order_documents_topics.csv";
    string metaPath = inputFolder + "/order_documents_meta.csv";

    cout<<"Processing order_promoted_content.csv"<<endl;
    vector<vector<string> > promotedVector = convertToVector(promotedPath);

    cout<<"Processing order_events.csv"<<endl;
    vector<vector<string> > eventVector = convertToVector(eventPath);

    cout<<"Process order_documents_categories.csv"<<endl;
    vector<vector<string> > docCategoriesVector = convertToVector(categoriesPath);

    cout<<"Process order_documents_topics.csv"<<endl;
    vector<vector<string> > docTopicVector = convertToVector(topicsPath);

    cout<<"Processing order_documents_meta.csv"<<endl;
    vector<vector<string> > docMetaVector = convertToVector(metaPath);

    cout<<"Start to catch feature..."<<endl;
    string row;
    getline(trainDataFile, row);
    int index = 0;
    while(getline(trainDataFile, row))
    {
        cout<<"===="<<index<<"===="<<endl;
        vector<string> rowVector = split(row, ',');
//        string label = rowVector[2];

        int document_id = -1;

        stringstream str_display_id(rowVector[0]);
        int display_id = -1;
        str_display_id >> display_id;

        stringstream str_doc_id_by_display(eventVector[display_id][2]);
        str_doc_id_by_display >> document_id;
        cout<<document_id<<endl;
        featuresFile<<display_id<<" "<<docCategoriesVector[document_id][1]<<" "<<docTopicVector[document_id][1]<<" "<<docMetaVector[document_id][1]<<" "<<docMetaVector[document_id][2]<<" ";

        stringstream str_ad_id(rowVector[1]);
        int ad_id = -1;
        str_ad_id >> ad_id;

        stringstream str_doc_id_by_ad(promotedVector[ad_id][1]);
        str_doc_id_by_ad >> document_id;
        cout<<document_id<<endl;
        featuresFile<< docCategoriesVector[document_id][1]<<" "<<docTopicVector[document_id][1]<<" "<<docMetaVector[document_id][1]<<" "<<docMetaVector[document_id][2]<<endl;

        index++;
    }

    trainDataFile.close();
    featuresFile.close();
    return 0;
}


