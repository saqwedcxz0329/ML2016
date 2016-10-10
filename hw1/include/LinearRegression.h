#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

class LinearRegression
{
    public:
        LinearRegression();
        virtual ~LinearRegression();
        vector<double> training(map<string, map<string, vector<double> > >);
        map<string, double> testResult(vector<double>,  map<string, map<string, vector<double> > > );
    private:
        vector<double> initX(int);
        double lossFunction(map<string, map<string, vector<double> > > , vector<double>& , vector<vector<double> >& );
        void gradientDescent(vector<double>&, vector<double>, vector<vector<double> >);
        double regularization(vector<double>, vector<double>&, double);

};

#endif // LINEARREGRESSION_H
