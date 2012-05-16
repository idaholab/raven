#include "inputFile.h"

void LoadData(double** data, int dimensionality, int cardinality, std::string filename) {
  int x, y;

  std::ifstream in(filename.c_str());

  if (!in) {
    std::cout << "Cannot open file.\n";
    return;
  }

  for (y = 0; y < cardinality; y++) {
    for (x = 0; x < dimensionality; x++) {
      in >> data[y][x];
    }
  }

  in.close();
}
