#ifndef CSV_READER_H
#define CSV_READER_H
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


//Helper function for loading trajectory data from ../Data/ into data_ptr
long load_data(std::string dataPath, float **&data_ptr, float *DT, const long & Nx)
{
    using namespace std;
    fstream f;
    std::cout << dataPath << std::endl;
    f.open(dataPath, ios::in);
    string line;
    getline(f, line);
    long lIterates = stol(line, NULL, 10);
    getline(f, line);
    stringstream s0(line);
    string word;
    string::size_type sz;
    DT[0] = stod(line, &sz);
    float **data = new float *[lIterates];
    for (int i = 0; i < lIterates; i++)
    {
        data[i] = new float[Nx];
    }
    for (long i = 0; i < lIterates; ++i)
    {

        getline(f, line);
        stringstream s(line);
        for (int k = 0; k < 3; k++)
        {
            getline(s, word, ',');
            data[i][k] = stod(word);
        }
    }
    f.close();
    data_ptr = data;
    return lIterates;
}
#endif