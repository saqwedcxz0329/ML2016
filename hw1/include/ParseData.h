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
        map<string, map<string, vector<double> > > captureData();

    private:
        ifstream data_file;
        vector<string> split(string, char*);
};

#endif // PARSEDATA_H
