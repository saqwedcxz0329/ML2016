#ifndef PARSEDATA_H
#define PARSEDATA_H
#include <iostream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <map>

using namespace std;


class ParseData
{
    public:
        ParseData();
        virtual ~ParseData();
        void readFile(char*);
        void outFile(char*, map<string, double>);
        map<string, map<string, vector<double> > > captureData();

    private:
        string filePath;
        ifstream data_file;
        ofstream data_output_file;
        vector<string> split(string, char*);
        bool isTraining();
};

#endif // PARSEDATA_H
