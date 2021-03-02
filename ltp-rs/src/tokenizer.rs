use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
pub use tokenizers::tokenizer::Tokenizer;
use tokenizers::tokenizer::{Model, PaddingDirection, PaddingParams, PaddingStrategy};

pub struct LTPTokenizer;

impl LTPTokenizer {
    pub fn new(path: &str) -> Tokenizer {
        let wordpiece_builder = WordPiece::from_files(path);
        let wordpiece = Box::new(wordpiece_builder.build().unwrap());
        // {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}

        let cls_token = String::from("[CLS]");
        let sep_token = String::from("[SEP]");
        let pad_token = String::from("[PAD]");
        let cls_id = wordpiece.token_to_id(cls_token.as_str()).unwrap();
        let sep_id = wordpiece.token_to_id(sep_token.as_str()).unwrap();
        let pad_id = wordpiece.token_to_id(pad_token.as_str()).unwrap();

        let sep = (sep_token, sep_id);
        let cls = (cls_token, cls_id);

        let bert_pretokenizer = Box::new(BertPreTokenizer);
        let bert_processor = Box::new(BertProcessing::new(sep, cls));
        let bert_normalizer = Box::new(BertNormalizer::default());
        let padding_params = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_id: pad_id,
            pad_type_id: 0,
            pad_token,
        };

        let mut tokenizer = Tokenizer::new(wordpiece);
        tokenizer.with_pre_tokenizer(bert_pretokenizer);
        tokenizer.with_post_processor(bert_processor);
        tokenizer.with_normalizer(bert_normalizer);
        tokenizer.with_padding(Option::from(padding_params));

        tokenizer
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::LTPTokenizer;
    use tokenizers::tokenizer::EncodeInput;

    #[test]
    fn test_tokenizer() {
        let tokenizer = LTPTokenizer::new("../vocab.txt");

        let input = String::from("他叫汤姆去拿外衣！");
        let input = EncodeInput::Single(input);

        let encoding = tokenizer.encode(input, true).unwrap();
        println!("{:?}", encoding.get_tokens());
        println!("{:?}", encoding.get_ids());
        println!("{:?}", encoding.get_type_ids());
        println!("{:?}", encoding.get_attention_mask());
        println!("{:?}", encoding.get_offsets());

        let input = vec![
            EncodeInput::Single(String::from("他叫汤姆去拿外衣！")),
            EncodeInput::Single(String::from("我爱中国")),
        ];

        let encodings = tokenizer.encode_batch(input, true).unwrap();
        println!("{:?}", encodings[1].get_tokens());
        println!("{:?}", encodings[1].get_ids());
        println!("{:?}", encodings[1].get_type_ids());
        println!("{:?}", encodings[1].get_attention_mask());
        println!("{:?}", encodings[1].get_offsets());
    }
}
