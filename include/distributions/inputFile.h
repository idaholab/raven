/*
 * InputFile.h
 *
 *  Created on: Oct 6, 2011
 *      Author: MANDD
 */

#ifndef INPUTFILE_H_
#define INPUTFILE_H_

#include <sstream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;

static void LoadData(double** data, int dimensionality, int cardinality, string filename) {
  int x, y;

  ifstream in(filename.c_str());

  if (!in) {
    cout << "Cannot open file.\n";
    return;
  }

  for (y = 0; y < cardinality; y++) {
    for (x = 0; x < dimensionality; x++) {
      in >> data[y][x];
    }
  }

  in.close();
}



#endif /* INPUTFILE_H_ */
