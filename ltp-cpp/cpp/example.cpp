#include <ltp.h>
#include <iostream>
using namespace std;

int main(int argc, char **argv) {
  auto ltp = ltp_init("onnx-small");
  auto sentences = rust::Vec<rust::String>{"他叫汤姆去拿外衣！"};
  auto results = ltp->pipeline(sentences);

  for (auto &result: results) {
    for (auto &word:result.seg()) cout << word.data() << "\t";
    cout << endl;

    for (auto &tag:result.pos()) cout << tag.data() << "\t";
    cout << endl;

    for (auto &tag:result.ner()) cout << tag.data() << "\t";
    cout << endl;

    auto word_num = result.len();
    for (auto i = 0; i < word_num; i++) {
      for (auto &srls:result.srl(i)) {
        cout << srls.data() << "\t";
      }
      cout << endl;
    }

    for (auto i = 0; i < word_num; i++) {
      cout << result.dep_arc(i) << ":" << result.dep_rel(i).data() << "\t";
    }
    cout << endl;

    auto sdp_num = result.sdp_len();
    for (auto i = 0; i < sdp_num; i++) {
      cout << result.sdp_src(i) << ":" << result.sdp_tgt(i) << ":" << result.sdp_rel(i).data() << "\t";
    }
    cout << endl;
  }
  return 0;
}


