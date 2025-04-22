#include "ModelManager/ModelManager.h"
int main(int argc, char ** argv) {
  using namespace KernelCodeGen;
  ModelManager m;
  m.process(std::string(argv[1]));
  return 0;
}