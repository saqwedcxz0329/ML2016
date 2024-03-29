#include "LinearRegression.h"

LinearRegression::LinearRegression(vector<double> m_yheads)
{
    this->y_heads = m_yheads;
}

LinearRegression::~LinearRegression()
{
    //dtor
}

void LinearRegression::training(vector<vector<double> > train_set)
{
    int w_num = train_set[0].size();
    /***** 1 dimension *****/
    initParameters(w_num + 1); // initial b and wi
    /***** 2 dimension *****/
//    initParameters(w_num*2 + 1);
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

        double error_value = lossFunction(train_set, past_gradients);
        cout<<i<<"==="<<error_value<<endl;
        i++;
        if(i >= 10000)
        {
            break;
        }
//        if (nowTime.tm_hour - start_hour >= 7){
//            cout<<error_value<<endl;
//            break;
//        }
    }
    cout<<nowTime.tm_hour<<":"<<nowTime.tm_min<<":"<<nowTime.tm_sec<<endl;

}

map<string, double> LinearRegression::testResult(map<string, map<string, vector<double> > > test_data)
{
    map<string, double> predict_pm;
    for(map<string, map<string, vector<double> > >::iterator outer_iter = test_data.begin(); outer_iter != test_data.end(); ++outer_iter)
    {
        vector<double> features;
        features.push_back(1); // x0 = 1;
        for(map<string, vector<double> >::iterator inner_iter = outer_iter->second.begin(); inner_iter != outer_iter->second.end(); ++inner_iter )
        {
            string item = inner_iter->first;
            vector<double> values = inner_iter->second;
            /***** 1 dimension *****/
            features.insert(features.end(), values.begin()+4, values.end()); // add all column elements
            /***** 2 dimension *****/
//            if(item != "AMB_TEMP" && item != "RH" && item != "WIND_DIREC" && item!= "WIND_SPEED")
//            {
//                for(int i = 4; i < values.size(); i+=9)
//                {
//                    for(int j = i; j < i+5; j++)
//                    {
//                        features.push_back(values[j]);
//                        features.push_back(values[j] * values[j]);
//
//                    }
//                }
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


double LinearRegression::lossFunction(vector<vector<double> > train_set, vector<double*>  &past_gradients)
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
        /***** 1dimension *****/
        for(int i = 1; i < parameters_num; i++)
        {
            features[i] = train_set[m][i-1];

        }
        /***** 2dimension *****/
//        for(int i = 1; i < parameters_num; i+=2)
//        {
//
//            features[i] = train_set[m][(i-1)/2];
//            features[i+1] = train_set[m][(i-1)/2] * train_set[m][(i-1)/2];
//        }

        double y = 0;
        for(int i =0; i < parameters_num; i++)
        {

            y = y + parameters[i]*features[i];

        }

        for(int i = 0; i < parameters_num; i++)
        {
            gradients[i] = gradients[i] + (-2)*(y_head - y) * features[i]; //compute every feature's gradient
        }

        error_value = error_value + (y_head - y)*(y_head - y);
    }

    /***** regularization *****/
    double lambda = 100;
    if(lambda!=0)
    {
        double sigma_w_square = regularization(gradients, lambda);
        error_value = error_value + lambda*sigma_w_square;
//        cout<<sigma_w_square<<endl;
    }

    past_gradients.push_back(gradients);
    gradientDescent(gradients, past_gradients);

    return error_value;
}

void LinearRegression::gradientDescent(double gradients[], vector<double*> past_gradients)
{
    int gradients_num = parameters.size();
    double learning_rate = 1;
    double sigma_past[gradients_num] = {0};

    for(int i = 0; i < past_gradients.size(); i++)
    {
        for(int j = 0; j < gradients_num; j++)
        {
            sigma_past[j] = sigma_past[j] + past_gradients[i][j] * past_gradients[i][j];// Agagrad, compute the past gradient's sum
        }
    }


    for(int i = 0; i < parameters.size(); i++)
    {
        parameters[i] = parameters[i] - learning_rate * gradients[i] / sqrt(sigma_past[i]);// Adjust the weight
    }
}

double LinearRegression::regularization(double gradients[], double lambda)
{
    double sigma_w_square = 0;
    for(int i = 0 ; i < parameters.size(); i++)
    {
        sigma_w_square = sigma_w_square + parameters[i]*parameters[i];
        gradients[i] = gradients[i] + 2 * lambda * parameters[i];
    }
    return sigma_w_square;
}

void LinearRegression::initParameters(int feature_num)
{
    srand(time(NULL));
    double ran;
    for (int i=0; i<feature_num; i++)
    {
        ran = (((float)rand()/(float)(RAND_MAX)))  * 0.01;
//        ran = 0.001;
        parameters.push_back(ran);
    }
}
