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

void ParseData::readFile(char* filePath)
{

    data_file.open(filePath);
}

map<string, map<string, vector<double> > > ParseData::captureData()
{
    string label;
    getline(data_file, label, '\n');

    string data_info;
    map<string, map<string, vector<double> > > day_value;
    map<string, vector<double> > item_value;
    while(getline(data_file, data_info, '\n'))
    {
        vector<string> v = split(data_info, ",");
        vector<double> value;
        for(int i = 3; i < v.size(); i++)
        {
            double d = atof(v[i].c_str());
            value.push_back(d);
        }
        item_value[v[2]] = value;
        if(v[2] == "WS_HR"){
            day_value[v[0]] = item_value;
            map<string, vector<double> > item_value;
        }
    }
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
