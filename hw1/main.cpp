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

    parsedata.readFile("./data/own_train.csv");
    map<string, map<string, vector<double> > > train_data = parsedata.captureData();
    vector<double> y_heads;
    vector<vector<double> > train_set = parsedata.convertDS(train_data, y_heads);
    parsedata.readFile("./data/own_test_X.csv");
    map<string, map<string, vector<double> > > test_data = parsedata.captureData();

    LinearRegression linearregression(y_heads);
    linearregression.training(train_set);
    map<string, double> predict_pm = linearregression.testResult(test_data);

    parsedata.outFile("./linear_regression.csv", predict_pm);

    return 0;
}
