#include <iostream>
using namespace std;
extern "C" {
    int say_hello(){
        std::cout << "HelloWorld!" << std::endl;
        return 0;
    }
}