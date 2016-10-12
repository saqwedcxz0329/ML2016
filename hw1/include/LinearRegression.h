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
    LinearRegression();
    virtual ~LinearRegression();
    vector<double> training(vector<vector<double> >, vector<double>);
    map<string, double> testResult(vector<double>,  map<string, map<string, vector<double> > > );
private:
    vector<double> initW(int);
    double lossFunction(vector<vector<double> >, vector<double>&, vector<double*> &, vector<double>);
    void gradientDescent(vector<double> &, double [], int, vector<double*>  );
    double regularization(vector<double>, double *, int, double);

};

#endif // LINEARREGRESSION_H
