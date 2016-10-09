#include "LinearRegression.h"
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <math.h>

LinearRegression::LinearRegression()
{
    //ctor
}

LinearRegression::~LinearRegression()
{
    //dtor
}

vector<double> LinearRegression::training(map<string, map<string, vector<double> > > train_set)
{
    int x_num = train_set["2014/1/1"].size() * (train_set["2014/1/1"]["AMB_TEMP"].size()-1);
    vector<double> parameters = initX(x_num + 1); // initial b and wi
    vector<vector<double> > past_gradients;
    while(1)
    {
        double error_value = lossFunction(train_set, parameters, past_gradients);
        cout<<error_value<<endl;
        if (error_value<3800)break;
    }
    return parameters;

}

map<string, double> LinearRegression::testResult(vector<double> parameters, map<string, map<string, vector<double> > > test_data)
{
    map<string, double> predict_pm;
    for(map<string, map<string, vector<double> > >::iterator outer_iter = test_data.begin(); outer_iter != test_data.end(); ++outer_iter)
    {
        vector<double> features;
        features.push_back(1); // x0 = 1;
        for(map<string, vector<double> >::iterator inner_iter = outer_iter->second.begin(); inner_iter != outer_iter->second.end(); ++inner_iter )
        {
            vector<double> values = inner_iter->second;
            /***** 1 dimension *****/
            features.insert(features.end(), values.begin(), values.end()); // add the whole values
            /***** 2 dimension *****/
//            for(int i = 0; i < values.size(); i++){
//                features.push_back(values[i]);
//                features.push_back(values[i] * values[i]);
//            }
        }
        try
        {
            if (features.size()!=parameters.size())
            {
                cout<<"feature size: "<<features.size()<<" parameter size: "<<parameters.size()<<endl;
                throw "parameter number is not equal feature number(testResult)";
            }
        }
        catch(const char* message)
        {
            cout<<message<<endl;
            exit(0);
        }
        double y = 0;
        for(int i =0; i < features.size(); i++)
        {
            y = y + parameters[i]*features[i];
        }
//        cout<<outer_iter->first<<"  "<<y<<endl;
        predict_pm[outer_iter->first] = y;
    }
    return predict_pm;
}


double LinearRegression::lossFunction(map<string, map<string, vector<double> > > train_set, vector<double>& parameters, vector<vector<double> >& past_gradients)
{
    double error_value = 0;
    vector<double> gradients;

    for(int i = 0; i < parameters.size(); i++)
    {
        gradients.push_back(0);
    }

    for(map<string, map<string, vector<double> > >::iterator outer_iter = train_set.begin(); outer_iter != train_set.end(); ++outer_iter)
    {
        vector<double> tmp = outer_iter->second["PM2.5"];
        double y_head = tmp[tmp.size()-1];
        vector<double> features;
        features.push_back(1); // x0 = 1;
        for(map<string, vector<double> >::iterator inner_iter = outer_iter->second.begin(); inner_iter != outer_iter->second.end(); ++inner_iter )
        {
            vector<double> values = inner_iter->second;
            /***** 1 dimension *****/
            features.insert(features.end(), values.begin(), values.end()-1); // don't add the last column value
            /***** 2 dimension *****/
//            for(int i = 0; i < values.size()-1; i++){
//                features.push_back(values[i]);
//                features.push_back(values[i] * values[i]);
//            }
        }
        try
        {
            if (features.size()!=parameters.size())
            {
                cout<<"feature size: "<<features.size()<<" parameter size: "<<parameters.size()<<endl;
                throw "parameter number is not equal feature number(lossFunction)";
            }
        }
        catch(const char* message)
        {
            cout<<message<<endl;
            exit(0);
        }

        double y = 0;
        for(int i =0; i < features.size(); i++)
        {
            y = y + parameters[i]*features[i];
        }
        error_value = error_value + (y_head - y)*(y_head - y);

        for(int i = 0; i < parameters.size(); i++)
        {
            gradients[i] = gradients[i] + (-2)*(y_head - y) * features[i];
        }
    }
    past_gradients.push_back(gradients);
    gradientDescent(parameters, gradients, past_gradients);

    return error_value;
}

void LinearRegression::gradientDescent(vector<double>&parameters, vector<double>gradients, vector<vector<double> > past_gradients)
{
    double learning_rate = 0.1;
    vector<double> sigma_past;
    for(int i = 0; i < gradients.size(); i++) //change to gradients's size
    {
        sigma_past.push_back(0);
    }
    for(int i = 0; i < past_gradients.size(); i++)
    {
        for(int j = 0; j < past_gradients[i].size(); j++)
        {
            sigma_past[j] = sigma_past[j] + past_gradients[i][j] * past_gradients[i][j];
        }

    }

    try
    {
        if (sigma_past.size()!=parameters.size())
        {
            cout<<"sigma_past size: "<<sigma_past.size()<<" parameter size: "<<parameters.size()<<endl;
            throw "parameter number is not equal feature number(gradientDescent)";
        }
    }
    catch(const char* message)
    {
        cout<<message<<endl;
        exit(0);
    }


    for(int i = 0; i < parameters.size(); i++)
    {

        parameters[i] = parameters[i] - learning_rate*gradients[i] / sqrt(sigma_past[i]);
    }
}

vector<double> LinearRegression::initX(int feature_num)
{
    vector<double> parameters;
    srand(time(NULL));
    double ran;
    for (int i=0; i<feature_num; i++)
    {
        ran = (((float)rand()/(float)(RAND_MAX)) )  * 0.1;
        parameters.push_back(ran);
    }
    return parameters;
}
