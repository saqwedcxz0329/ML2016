#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <stdlib.h>
#include <ctime>
#include <math.h>

using namespace std;

class LinearRegression
{
public:
    LinearRegression(vector<double>);
    virtual ~LinearRegression();
    void training(vector<vector<double> >);
    map<string, double> testResult(map<string, map<string, vector<double> > > );
private:
    vector<double> parameters;
    vector<double> y_heads;
    void initParameters(int);
    double lossFunction(vector<vector<double> >, vector<double*> &);
    void gradientDescent(double [], int, vector<double*>);
    double regularization(double *, int, double);

};

#endif // LINEARREGRESSION_H
