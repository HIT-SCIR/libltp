#include <ltp.h>
#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <iostream>

int main(int argc, char **argv) {
  auto ltp = ltp::ltp_init("onnx-small");

  arrow::StringBuilder sentences_builder{};
  std::shared_ptr<arrow::Array> sentences_array;

  sentences_builder.Append("他叫汤姆去拿外衣。");
  sentences_builder.Append("我爱赛尔。");
  sentences_builder.Finish(&sentences_array);

  ArrowArray arrow_array{};
  ArrowSchema arrow_schema{};

  arrow::ExportArray(*sentences_array, &arrow_array, &arrow_schema);

  ltp::LTPResultArray result_array{};

  const ArrowArray *arrow_array_stack[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  const ArrowSchema *arrow_schema_stack[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  result_array.Arrays = arrow_array_stack;
  result_array.Schemes = arrow_schema_stack;

  ltp::ltp_pipeline(ltp, &arrow_array, &arrow_schema, &result_array);

  auto seg_array_list_list = arrow::ImportArray((ArrowArray *) arrow_array_stack[0],
                                                (ArrowSchema *) arrow_schema_stack[0]).ValueOrDie();

  auto pos_array_list_list = arrow::ImportArray((ArrowArray *) arrow_array_stack[1],
                                                (ArrowSchema *) arrow_schema_stack[1]).ValueOrDie();

  auto ner_array_list_list = arrow::ImportArray((ArrowArray *) arrow_array_stack[2],
                                                (ArrowSchema *) arrow_schema_stack[2]).ValueOrDie();

  auto srl_array_list_list = arrow::ImportArray((ArrowArray *) arrow_array_stack[3],
                                                (ArrowSchema *) arrow_schema_stack[3]).ValueOrDie();

  auto dep_array_list_list = arrow::ImportArray((ArrowArray *) arrow_array_stack[4],
                                                (ArrowSchema *) arrow_schema_stack[4]).ValueOrDie();

  auto sdp_array_list_list = arrow::ImportArray((ArrowArray *) arrow_array_stack[5],
                                                (ArrowSchema *) arrow_schema_stack[5]).ValueOrDie();

  //  std::cout << *seg_array_list_list << std::endl;
  //  std::cout << *pos_array_list_list << std::endl;
  //  std::cout << *ner_array_list_list << std::endl;
  //  std::cout << *srl_array_list_list << std::endl;
  //  std::cout << *dep_array_list_list << std::endl;
  //  std::cout << *sdp_array_list_list << std::endl;

  // how to get inner string
  auto seg_first_sentence = std::static_pointer_cast<arrow::ListScalar>(seg_array_list_list->GetScalar(0).ValueOrDie());
  auto seg_first_sentence_value = std::static_pointer_cast<arrow::StringArray>(seg_first_sentence->value);

  for (int i = 0; i < seg_first_sentence_value->length(); ++i) {
    std::cout << seg_first_sentence_value->GetString(i) << std::endl;
  }

  // how to get inner string
  auto sdp_first_sentence = std::static_pointer_cast<arrow::ListScalar>(sdp_array_list_list->GetScalar(0).ValueOrDie());
  auto sdp_first_sentence_value = std::static_pointer_cast<arrow::StructArray>(sdp_first_sentence->value);

  for (int i = 0; i < sdp_first_sentence_value->length(); ++i) {
    auto sdp_first_sentence_value_struct =
        std::static_pointer_cast<arrow::StructScalar>(sdp_first_sentence_value->GetScalar(i).ValueOrDie());
    auto values = sdp_first_sentence_value_struct->value;
    uint64_t src = std::static_pointer_cast<arrow::UInt64Scalar>(values[0])->value;
    uint64_t tgt = std::static_pointer_cast<arrow::UInt64Scalar>(values[1])->value;
    std::shared_ptr<arrow::Buffer> rel = std::static_pointer_cast<arrow::StringScalar>(values[2])->value;

    auto rel_view = static_cast<arrow::util::string_view>(*rel);
    std::cout << src << " -> " << tgt << " : " << rel_view << std::endl;
  }

  ltp::ltp_release(ltp);
  return 0;
}


