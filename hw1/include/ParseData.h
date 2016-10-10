#ifndef PARSEDATA_H
#define PARSEDATA_H
#include <iostream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <map>
#include <math.h>

using namespace std;


class ParseData
{
    public:
        ParseData();
        virtual ~ParseData();
        void readFile(char*);
        void outFile(char*, map<string, double>);
        map<string, map<string, vector<double> > > captureData();
        vector<vector<double> > convertToTrainSet(map<string, map<string, vector<double> > >, vector<double>&);

    private:
        string filePath;
        ifstream data_file;
        ofstream data_output_file;
        vector<string> split(string, char*);
        void featureScaling(vector<vector<double> >& );
        bool isTraining();
};

#endif // PARSEDATA_H
