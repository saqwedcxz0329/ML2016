#include "ParseData.h"

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
        if(predict_pm[id]<0)
        {
            predict_pm[id]=0;
        }
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

vector<vector<double> >  ParseData::convertDS(map<string, map<string, vector<double> > > datas, vector<double> &y_heads)
{
    vector<vector<double> > train_set;
    for(map<string, map<string, vector<double> > >::iterator outer_iter = datas.begin(); outer_iter != datas.end(); ++outer_iter)
    {
        int value_size = outer_iter->second["PM2.5"].size();
        y_heads.push_back(outer_iter->second["PM2.5"][value_size-1]);
        vector<double> features;
        for(map<string, vector<double> >::iterator inner_iter = outer_iter->second.begin(); inner_iter != outer_iter->second.end(); ++inner_iter )
        {
            string item = inner_iter->first;
//            if(item != "AMB_TEMP" && item != "RH" && item != "WIND_DIREC" && item!= "WIND_SPEED")
//            {
            vector<double> values = inner_iter->second;
            features.insert(features.end(), values.begin(), values.end()-1); // don't add the last column value
//            }
        }
        train_set.push_back(features);
    }

//    featureScaling(train_set);
//    for(int i = 0; i < train_set[0].size(); i++){
//        cout<<train_set[0][i]<<endl;
//    }

    return train_set;
}

vector<vector<double> > ParseData::convertDS(map<string, map<string, vector<double> > > datas)
{
    vector<vector<double> > test_set;
    for(map<string, map<string, vector<double> > >::iterator outer_iter = datas.begin(); outer_iter != datas.end(); ++outer_iter)
    {
        vector<double> features;
        for(map<string, vector<double> >::iterator inner_iter = outer_iter->second.begin(); inner_iter != outer_iter->second.end(); ++inner_iter )
        {
            string item = inner_iter->first;
//            if(item != "AMB_TEMP" && item != "RH" && item != "WIND_DIREC" && item!= "WIND_SPEED")
//            {
            vector<double> values = inner_iter->second;
            features.insert(features.end(), values.begin(), values.end()); // add whole elements
//            }
        }
        test_set.push_back(features);
    }

//    featureScaling(test_set);
//    for(int i = 0; i < test_set[0].size(); i++){
//        cout<<test_set[0][i]<<endl;
//    }

    return test_set;
}


void ParseData::featureScaling(vector<vector<double> > &train_set)
{
    for(int i = 0; i < train_set.size(); i++)
    {
        double sigma_x = 0;
        double sigma_x_square = 0;
        for(int j = 0; j < train_set[i].size(); j++)
        {
            sigma_x = sigma_x + train_set[i][j];
            sigma_x_square = sigma_x_square + train_set[i][j] * train_set[i][j];
        }
        double mean_x = 0;
        double sd_x = 0;
        mean_x = sigma_x / train_set[i].size();
        sd_x = sqrt( sigma_x_square / train_set[i].size()  - mean_x*mean_x);

        for(int j = 0; j < train_set[i].size(); j++)
        {
            train_set[i][j] = (train_set[i][j] - mean_x) / sd_x;
        }
    }
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
