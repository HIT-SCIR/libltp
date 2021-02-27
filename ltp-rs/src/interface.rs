use std::ffi::OsStr;
use std::path::Path;

use tokenizers::tokenizer::EncodeInput;
use itertools::{Itertools, multizip};
use std::borrow::{Borrow};
use lazy_static::lazy_static;

use onnxruntime as onnx;
use onnxruntime::environment::{Environment};
use onnxruntime::ndarray::{ArrayBase, prelude::s};
use onnxruntime::session::Session;
use onnxruntime::GraphOptimizationLevel;
use onnxruntime::tensor::OrtOwnedTensor;
use serde::{Deserialize, Serialize};

use crate::vocabs::Vocab;
use crate::tokenizer::{LTPTokenizer, Tokenizer};

use crate::eisner::eisner;
use crate::entities::get_entities;
use crate::viterbi::viterbi_decode_postprocess;

use crate::Result;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DEP {
    pub arc: usize,
    pub rel: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SDP {
    pub src: usize,
    pub tgt: usize,
    pub rel: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LTPResult {
    pub seg: Option<Vec<String>>,
    pub pos: Option<Vec<String>>,
    pub ner: Option<Vec<String>>,
    pub dep: Option<Vec<DEP>>,
    pub sdp: Option<Vec<SDP>>,
    pub srl: Option<Vec<Vec<String>>>,
}

lazy_static! {
    static ref G_ENV : Environment = Environment::builder()
            .with_name("ltp")
            // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
            .with_log_level(onnx::LoggingLevel::Info)
            .build().unwrap();
}

pub struct LTP {
    vocabs: Vocab,
    tokenizer: Tokenizer,
    session: Session<'static>,
}

macro_rules! option_vec_to_vec_option {
    ($option_items:expr, $size:expr) => {
        $option_items.map_or(
            vec![None; $size],
            |items| items.into_iter().map(|item| Some(item)).collect_vec(),
        )
    };
}

impl LTP {
    pub fn new<S: AsRef<OsStr> + ?Sized>(path: &S) -> Result<LTP> {
        LTP::new_with_options(path, GraphOptimizationLevel::All, 1)
    }

    pub fn new_with_options<S: AsRef<OsStr> + ?Sized>(path: &S, optimization_level: onnx::GraphOptimizationLevel, num_threads: i16) -> Result<LTP> {
        let vocabs = Vocab::load(Path::new(path).join("vocab.json").to_str().unwrap())?;
        let tokenizer = LTPTokenizer::new(Path::new(path).join("vocab.txt").to_str().unwrap());
        let onnx = String::from(Path::new(path).join("ltp.onnx").to_str().unwrap());

        let session = G_ENV
            .new_session_builder()?
            .with_optimization_level(optimization_level)?
            .with_number_threads(num_threads)?
            .with_model_from_file(onnx)?;

        Ok(LTP { vocabs, tokenizer, session })
    }

    #[cfg(feature = "coreml")]
    pub fn new_with_coreml_options<S: AsRef<OsStr> + ?Sized>(path: &S, optimization_level: onnx::GraphOptimizationLevel, num_threads: i16, flags: OnnxEnumInt) -> Result<LTP> {
        let vocabs = Vocab::load(Path::new(path).join("vocab.json").to_str().unwrap())?;
        let tokenizer = LTPTokenizer::new(Path::new(path).join("vocab.txt").to_str().unwrap());
        let onnx = String::from(Path::new(path).join("ltp.onnx").to_str().unwrap());

        let session = G_ENV
            .new_session_builder()?
            .with_optimization_level(optimization_level)?
            .with_number_threads(num_threads)?
            .with_coreml(flags)?
            .with_model_from_file(onnx)?;

        Ok(LTP { vocabs, tokenizer, session })
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_cuda_options<S: AsRef<OsStr> + ?Sized>(path: &S, optimization_level: onnx::GraphOptimizationLevel, num_threads: i16, device_id: i32) -> Result<LTP> {
        let vocabs = Vocab::load(Path::new(path).join("vocab.json").to_str().unwrap())?;
        let tokenizer = LTPTokenizer::new(Path::new(path).join("vocab.txt").to_str().unwrap());
        let onnx = String::from(Path::new(path).join("ltp.onnx").to_str().unwrap());

        let session = G_ENV
            .new_session_builder()?
            .with_optimization_level(optimization_level)?
            .with_number_threads(num_threads)?
            .with_cuda(device_id)?
            .with_model_from_file(onnx)?;

        Ok(LTP { vocabs, tokenizer, session })
    }

    pub fn pipeline_batch(&mut self, sentences: &Vec<String>) -> Result<Vec<LTPResult>> {
        let inputs = sentences.iter().map(
            |s| EncodeInput::Single(s.to_string())
        ).collect_vec();
        let encodings = self.tokenizer.encode_batch(inputs, true).unwrap();
        let batch_size = encodings.len();

        let sentence_length = encodings[0].get_ids().len();

        let offsets = encodings.iter().map(|x| x.get_offsets()).collect_vec();
        let input_ids: Vec<i64> = encodings.iter().flat_map(|e| e.get_ids()).map(|&id| id as i64).collect();
        let token_type_ids: Vec<i64> = encodings.iter().flat_map(|e| e.get_type_ids()).map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encodings.iter().flat_map(|e| e.get_attention_mask()).map(|&id| id as i64).collect();
        let position_ids: Vec<i64> = vec![(0..sentence_length).collect_vec(); batch_size].iter().flatten().map(|&x| x as i64).collect_vec();

        let sentence_lengths: Vec<usize> = encodings.iter().map(
            |x| {
                let len: usize = x.get_attention_mask().iter().map(|&x| x as usize).sum();
                let len = len - 2;
                len
            }
        ).collect_vec();

        let input_arrays = vec![
            ArrayBase::from_shape_vec((batch_size, sentence_length), input_ids)?,
            ArrayBase::from_shape_vec((batch_size, sentence_length), token_type_ids)?,
            ArrayBase::from_shape_vec((batch_size, sentence_length), attention_mask)?,
            ArrayBase::from_shape_vec((batch_size, sentence_length), position_ids)?,
        ];

        let mut result = self.session.run(input_arrays)?;

        let seg = result.remove(0);
        let seg: OrtOwnedTensor<i64, _> = seg.try_extract()?;

        let seg_entities: Option<Vec<Vec<(&str, usize, usize)>>> = match self.vocabs.seg.as_ref() {
            Some(vocab) => Some({
                (0..batch_size).into_iter().zip(sentence_lengths).map(
                    |(idx, length)| {
                        get_entities(
                            seg.slice(s![idx,..length]).iter().map(|&x| vocab[x as usize].as_str()).collect()
                        )
                    }
                ).collect_vec()
            }),
            None => None,
        };

        let seg_entities: Option<Vec<Vec<(usize, usize)>>> = seg_entities.map(
            |x| x.iter().enumerate().map(
                |(idx, sent)| sent.iter().map(
                    |&(_tag, start, end)| (offsets[idx][start + 1].0, offsets[idx][end + 1].1)
                ).collect_vec()
            ).collect()
        );

        let texts = sentences.iter().map(|x| x.chars().collect_vec()).collect_vec();
        let texts: Option<Vec<Vec<String>>> = seg_entities.map(
            |x| x.iter().zip(texts).map(
                |(sent, text)|
                    sent.iter().map(|&(start, end)| text[start..end].iter().collect()).collect_vec()
            ).collect_vec()
        );

        let word_nums = texts.as_ref().map(
            |x| { x.iter().map(|sent| sent.len()).collect_vec() }
        ).unwrap();


        let pos: Option<Vec<Vec<String>>> = match self.vocabs.pos.as_ref() {
            Some(vocab) => Some({
                let pos = result.remove(0);
                let pos: OrtOwnedTensor<i64, _> = pos.try_extract()?;
                (0..batch_size).into_iter().zip(&word_nums).map(
                    |(idx, length)| {
                        pos.slice(s![idx,..*length]).iter().map(|&x| vocab[x as usize].clone()).collect()
                    }
                ).collect_vec()
            }),
            None => None,
        };

        let ner: Option<Vec<Vec<String>>> = match self.vocabs.ner.as_ref() {
            Some(vocab) => Some({
                let ner = result.remove(0);
                let ner: OrtOwnedTensor<i64, _> = ner.try_extract()?;
                (0..batch_size).into_iter().zip(&word_nums).map(
                    |(idx, length)| {
                        ner.slice(s![idx,..*length]).iter().map(|&x| vocab[x as usize].clone()).collect()
                    }
                ).collect_vec()
            }),
            None => None,
        };

        let srl = match self.vocabs.srl.as_ref() {
            Some(vocab) => Some({
                let srl_history: OrtOwnedTensor<i64, _> = result.remove(0).try_extract()?;
                let srl_last_tags: OrtOwnedTensor<i64, _> = result.remove(0).try_extract()?;
                viterbi_decode_postprocess(
                    srl_history.as_slice().unwrap(),
                    srl_last_tags.as_slice().unwrap(),
                    word_nums.as_slice(),
                    vocab.len(),
                ).iter().map(
                    |stn_srl| stn_srl.iter().map(
                        |tag| vocab[*tag as usize].clone()
                    ).collect_vec()
                ).collect_vec()
            }),
            None => None,
        };

        let srl = srl.map(|mut all_srl| {
            let mut results = Vec::new();
            for stn_len in &word_nums {
                let mut result = Vec::new();
                for _i in 0..*stn_len {
                    result.push(all_srl.remove(0))
                }
                results.push(result)
            }
            results
        });

        let cls_word_num = word_nums.iter().map(|&x| x + 1).collect_vec();
        let &max_cls_stn_length = cls_word_num.iter().max().unwrap();

        let dep = match self.vocabs.dep.as_ref() {
            Some(vocab) => Some({
                let dep_head: OrtOwnedTensor<f32, _> = result.remove(0).try_extract()?;
                let dep_labels: OrtOwnedTensor<i64, _> = result.remove(0).try_extract()?;
                let mut dep_head_decoded = vec![0usize; batch_size * max_cls_stn_length];
                eisner(
                    dep_head.as_slice().unwrap(),
                    cls_word_num.as_slice(),
                    &mut dep_head_decoded,
                );
                dep_head_decoded.iter().enumerate().map(
                    |(idx, head)| {
                        let batch = idx / max_cls_stn_length;
                        let word_idx = idx % max_cls_stn_length;

                        if word_idx < 1 || word_idx > word_nums[batch] { // 这里要考虑虚节点，所以不是 >=
                            None
                        } else {
                            Some(DEP {
                                arc: *head as usize,
                                rel: vocab[*(dep_labels.borrow().get([batch, word_idx, *head]).unwrap()) as usize].clone(),
                            })
                        }
                    }
                ).filter(|x| x.is_some()).map(|x| x.unwrap()).collect_vec()
            }),
            None => None,
        };

        let dep = dep.map(|mut all_dep| {
            let mut results = Vec::new();
            for stn_len in &word_nums {
                let mut result = Vec::new();
                for _i in 0..*stn_len {
                    result.push(all_dep.remove(0));
                }
                results.push(result)
            }
            results
        });

        let block_size = max_cls_stn_length * max_cls_stn_length;
        let sdp = match self.vocabs.sdp.as_ref() {
            Some(vocab) => Some({
                let sdp_head: OrtOwnedTensor<f32, _> = result.remove(0).try_extract()?;
                let sdp_labels: OrtOwnedTensor<i64, _> = result.remove(0).try_extract()?;

                let mut sdp_head_decoded = vec![0usize; batch_size * max_cls_stn_length];
                eisner(
                    sdp_head.as_slice().unwrap(),
                    cls_word_num.as_slice(),
                    &mut sdp_head_decoded,
                );
                sdp_head.iter().enumerate().map(
                    |(idx, score)| {
                        let batch = idx / block_size;
                        let block_idx = idx % block_size; // Z 字形

                        let current = block_idx / max_cls_stn_length;
                        let target = block_idx % max_cls_stn_length;
                        if current < 1 || current > word_nums[batch] || target > word_nums[batch] { // 这里要考虑虚节点
                            None
                        } else {
                            if *score > 0f32 || sdp_head_decoded[batch * max_cls_stn_length + current] == target {
                                let label = vocab[*(sdp_labels.borrow().get([batch, current, target]).unwrap()) as usize].clone();
                                return Some((batch, current, target, label));
                            }
                            None
                        }
                    }
                ).filter(|x| x.is_some()).map(|x| x.unwrap()).collect_vec()
            }),
            None => None,
        };

        let sdp = sdp.map(|all_sdp| {
            let mut results = vec![Vec::new(); batch_size];

            for (batch, current, target, tag) in all_sdp {
                results[batch].push(SDP { src: current, tgt: target, rel: tag });
            }
            results
        });

        let texts = option_vec_to_vec_option!(texts, batch_size);
        let pos = option_vec_to_vec_option!(pos, batch_size);
        let ner = option_vec_to_vec_option!(ner, batch_size);
        let srl = option_vec_to_vec_option!(srl, batch_size);
        let dep = option_vec_to_vec_option!(dep, batch_size);
        let sdp = option_vec_to_vec_option!(sdp, batch_size);

        let results = multizip((texts, pos, ner, srl, dep, sdp)).into_iter().map(
            |(
                 one_seg,
                 one_pos,
                 one_ner,
                 one_srl,
                 one_dep,
                 one_sdp
             )|
                LTPResult {
                    seg: one_seg,
                    pos: one_pos,
                    ner: one_ner,
                    dep: one_dep,
                    sdp: one_sdp,
                    srl: one_srl,
                }
        ).collect_vec();
        Ok(results)
    }

    pub fn pipeline(&mut self, sentence: &String) -> Result<LTPResult> {
        let fake_batch = vec![sentence.clone()];
        let results = self.pipeline_batch(&fake_batch);
        results.map(
            |mut result| result.remove(0)
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{LTP, LTPError};

    #[test]
    fn test_interface() -> Result<(), LTPError> {
        let path = String::from("../onnx-small");
        let mut ltp = LTP::new(&path)?;

        let sentence = String::from("他叫汤姆去拿外衣。");
        let result = ltp.pipeline(&sentence)?;

        let j = serde_json::to_string(&result).unwrap();
        println!("{}", j);

        let sentence2 = String::from("我爱赛尔!");
        let sentences = vec![sentence, sentence2];
        let result = ltp.pipeline_batch(&sentences)?;

        let j = serde_json::to_string(&result).unwrap();
        println!("{}", j);
        Ok(())
    }
}
