#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <StationDatas.h>
#include <ParseData.h>
#include <LinearRegression.h>

using namespace std;

int main()
{
    ParseData parsedata;
    LinearRegression linearregression;
    parsedata.readFile("./data/train_1.csv");
    map<string, map<string, vector<double> > > train_data = parsedata.captureData();
    parsedata.readFile("./data/test_X.csv");
    map<string, map<string, vector<double> > > test_data = parsedata.captureData();

    vector<double> parameters = linearregression.training(train_data);
    map<string, double> predict_pm = linearregression.testResult(parameters, test_data);

    parsedata.outFile("./data/predict.csv", predict_pm);

//    map<string, vector<double> > item_value = test_data["id_0"];
//    vector<double> value = item_value["CO"];
//    cout<<test_data.size()<<endl;
//    for(int i = 0; i < value.size(); i++)
//    {
//        cout<<value[i]<<endl;
//    }
    return 0;
}
