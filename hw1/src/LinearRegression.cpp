#include "LinearRegression.h"

LinearRegression::LinearRegression()
{
    //ctor
}

LinearRegression::~LinearRegression()
{
    //dtor
}

vector<double> LinearRegression::training(vector<vector<double> > train_set, vector<double> y_heads)
{
    int w_num = train_set[0].size();
    /***** 1 dimension *****/
    vector<double> parameters = initW(w_num + 1); // initial b and wi
    /***** 2 dimension *****/
//    vector<double> parameters = initW(x_num*2 + 1);
    vector<double*>  past_gradients;
    time_t now;
    struct tm nowTime;
    now = time(NULL);
    nowTime = *localtime(&now);
    int start_hour = nowTime.tm_hour;

    int i = 1;
    while(1)
    {
        now = time(NULL);
        nowTime = *localtime(&now);

        double error_value = lossFunction(train_set, parameters, past_gradients, y_heads);
        cout<<i<<"==="<<error_value<<endl;
        i++;
        if(i >= 75000){
            break;
        }
//        if (nowTime.tm_hour - start_hour >= 3){
//            cout<<error_value<<endl;
//            break;
//        }
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
            string item = inner_iter->first;
            /***** 1 dimension *****/
            if(item != "AMB_TEMP" && item != "RH" && item != "WIND_DIREC" && item!= "WIND_SPEED"){
                vector<double> values = inner_iter->second;
                features.insert(features.end(), values.begin(), values.end()); // add all column elements
            }
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
        predict_pm[outer_iter->first] = y;
    }
    return predict_pm;
}


double LinearRegression::lossFunction(vector<vector<double> > train_set, vector<double> &parameters, vector<double*>  &past_gradients, vector<double> y_heads)
{
    double error_value = 0;
    int parameters_num = parameters.size();
    double *gradients = new double [parameters_num];
    for(int i = 0; i < parameters_num; i++)
    {
        gradients[i] = 0;
    }
    for(int m = 0; m < train_set.size(); m++)
    {
        double y_head = y_heads[m];
        double features[parameters_num] = {0};

        features[0] = 1;
        for(int i = 1; i < parameters_num; i++)
        {
            features[i] = train_set[m][i-1];
        }

        double y = 0;
        for(int i =0; i < parameters_num; i++)
        {

            y = y + parameters[i]*features[i];

        }

        for(int i = 0; i < parameters_num; i++)
        {
            gradients[i] = gradients[i] + (-2)*(y_head - y) * features[i];
        }

        error_value = error_value + (y_head - y)*(y_head - y);
    }

    /***** regularization *****/
    double lambda = 0;
    if(lambda!=0)
    {
        double sigma_w_square = regularization(parameters, gradients, parameters_num, lambda);
        error_value = error_value + lambda*sigma_w_square;
    }

    past_gradients.push_back(gradients);
    gradientDescent(parameters, gradients, parameters_num, past_gradients);

    return error_value;
}

void LinearRegression::gradientDescent(vector<double> &parameters, double gradients[], int gradients_num, vector<double*> past_gradients)
{
    double learning_rate = 0.1;
    double sigma_past[gradients_num] = {0};
    int tmp_size = past_gradients.size();

    for(int i = 0; i < past_gradients.size(); i++)
    {
        for(int j = 0; j < gradients_num; j++)
        {
            sigma_past[j] = sigma_past[j] + past_gradients[i][j] * past_gradients[i][j];
        }
    }


    for(int i = 0; i < parameters.size(); i++)
    {
        parameters[i] = parameters[i] - learning_rate * gradients[i] / sqrt(sigma_past[i]);
    }
}

double LinearRegression::regularization(vector<double> parameters, double gradients[], int gradients_num, double lambda)
{
    double sigma_w_square = 0;
    double sigma_w = 0;
    for(int i = 0 ; i < parameters.size(); i++)
    {
        sigma_w_square = sigma_w_square + parameters[i]*parameters[i];
        sigma_w = sigma_w + parameters[i];
    }

    for(int i = 0; i < gradients_num; i++)
    {
        gradients[i] = gradients[i] + 2 * lambda * sigma_w;
    }
    return sigma_w_square;
}



vector<double> LinearRegression::initW(int feature_num)
{
    vector<double> parameters;
    srand(time(NULL));
    double ran;
    for (int i=0; i<feature_num; i++)
    {
        ran = (((float)rand()/(float)(RAND_MAX)))  * 0.01;
        parameters.push_back(ran);
    }
    return parameters;
}
