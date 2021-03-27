#include <ltp.h>
#include <cxx.h>
#include <iostream>
#include "argparse/argparse.hpp"
using namespace std;

int main(int argc, char **argv) {
  argparse::ArgumentParser program("ltp");

  program.add_argument("-m", "--model")
      .required()
      .help("model path");

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }
  auto model_path = program.get<string>("--model");
  try {
    auto ltp = ltp_init(model_path, 1, 0);
    auto sentences = rust::Vec<rust::String>{"他叫汤姆去拿外衣！"};
    auto results = ltp->pipeline(sentences);

    for (auto &result: results) {
      for (auto &word:result.seg()) cout << word << "\t";
      cout << endl;

      for (auto &tag:result.pos()) cout << tag << "\t";
      cout << endl;

      for (auto &tag:result.ner()) cout << tag << "\t";
      cout << endl;

      auto word_num = result.len();
      for (auto i = 0; i < word_num; i++) {
        for (auto &srls:result.srl(i)) {
          cout << srls << "\t";
        }
        cout << endl;
      }

      for (auto i = 0; i < word_num; i++) {
        cout << result.dep_arc(i) << ":" << result.dep_rel(i) << "\t";
      }
      cout << endl;

      auto sdp_num = result.sdp_len();
      for (auto i = 0; i < sdp_num; i++) {
        cout << result.sdp_src(i) << ":" << result.sdp_tgt(i) << ":" << result.sdp_rel(i) << "\t";
      }
      cout << endl;
    }
  } catch (rust::Error &e) {
    cerr << e.what() << endl;
  } catch (exception &e) {
    std::cerr << "Unknown failure occurred. Possible memory corruption." << e.what() << std::endl;
  }
  return 0;
}


