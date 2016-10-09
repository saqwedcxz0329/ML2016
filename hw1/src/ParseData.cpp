#include "ParseData.h"
#include <iostream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <map>

ParseData::ParseData()
{
    //ctor
}

ParseData::~ParseData()
{
    //dtor
}

void ParseData::readFile(char* m_filePath)
{
    string tmp(m_filePath);
    filePath = tmp;
    data_file.open(m_filePath);
}

void ParseData::outFile(char* filePath, map<string, double> predict_pm)
{
    data_output_file.open(filePath);
    data_output_file<<"id,value\n";
    char s[10];
    for(int i = 0; i < predict_pm.size(); i++)
    {
        sprintf(s, "%d", i);
        string tmp(s);
        string id = "id_" + tmp;
//        cout<<id<<endl;
        data_output_file<<id<<","<<predict_pm[id]<<"\n";
    }
}


map<string, map<string, vector<double> > > ParseData::captureData()
{
    string label;
    bool TrainingFlag = isTraining();
    int index = 2;
    if(TrainingFlag)
    {
        getline(data_file, label, '\n');
        index = 3;
    }

    string data_info;
    map<string, map<string, vector<double> > > day_value;
    map<string, vector<double> > item_value;
    while(getline(data_file, data_info, '\n'))
    {

        vector<string> v = split(data_info, ",");
        vector<double> value;
        for(int i = index; i < v.size(); i++)
        {
            double d = atof(v[i].c_str());
            value.push_back(d);
        }
        item_value[v[index-1]] = value;
        if(v[index-1] == "WS_HR")
        {
            day_value[v[0]] = item_value;
            map<string, vector<double> > item_value;
        }
    }
    data_file.close();
    return day_value;
}

vector<string> ParseData::split(string str, char* symbol)
{
    char* tmp = new char[str.length()+1];
    strcpy(tmp, str.c_str());

    vector<string> v;
    char* pch = NULL;
    pch = strtok (tmp,symbol);
    string str_pch(pch);
    v.push_back(str_pch);
    while(1)
    {
        pch = strtok (NULL, symbol);
        if(pch == NULL)
        {
            break;
        }
        string str_pch(pch);
        v.push_back(str_pch);
    }
    delete tmp;
    return v;
}

bool ParseData::isTraining()
{
    if(filePath.find("train") != string::npos)
    {
        return true;
    }
    return false;
}
