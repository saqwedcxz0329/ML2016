#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ParseData.h>
#include <LinearRegression.h>

using namespace std;

int main()
{
    ParseData parsedata;
    LinearRegression linearregression;
    parsedata.readFile("./data/train_1.csv");
    map<string, map<string, vector<double> > > train_data = parsedata.captureData();
    vector<double> y_heads;
    vector<vector<double> > train_set = parsedata.convertToTrainSet(train_data, y_heads);
    parsedata.readFile("./data/test_X.csv");
    map<string, map<string, vector<double> > > test_data = parsedata.captureData();

    vector<double> parameters = linearregression.training(train_set, y_heads);
    map<string, double> predict_pm = linearregression.testResult(parameters, test_data);

    parsedata.outFile("./data/predict.csv", predict_pm);

    return 0;
}
