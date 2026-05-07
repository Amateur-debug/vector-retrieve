#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "coromem/include/utils.h"

int main() {
  try {
    Logger logger("log.csv");

    logger << "Name" << "Age" << "Country";
    logger.end();
    logger << "Alice" << 30 << "USA";
    logger.end();
    logger << "Bob" << 25 << "UK";
    logger.end();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}
