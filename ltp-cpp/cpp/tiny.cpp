#include <ltp.h>
#include <cxx.h>
using namespace std;

int main(int argc, char **argv) {
  auto ltp = ltp_init("path/to/model", 1, 0);
  auto sentences = rust::Vec<rust::String>{"他叫汤姆去拿外衣！"};
  auto results = ltp->pipeline(sentences);
  return 0;
}


