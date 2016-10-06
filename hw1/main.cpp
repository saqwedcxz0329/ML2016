#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <StationDatas.h>
#include <ParseData.h>
typedef char* charPtr;
using namespace std;

int main()
{
    ParseData parsedata;
    parsedata.readFile("./data/train.csv");
    map<string, map<string, vector<double> > > day_value = parsedata.captureData();
    map<string, vector<double> > item_value = day_value["2014/12/19"];
    vector<double> value = item_value["CO"];
    cout<<day_value.size()<<endl;
    for(int i = 0; i < value.size(); i++)
    {
        cout<<value[i]<<endl;
    }
    return 0;
}
