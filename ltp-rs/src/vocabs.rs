use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize)]
pub struct Vocab {
    pub seg: Option<Vec<String>>,
    pub pos: Option<Vec<String>>,
    pub ner: Option<Vec<String>>,
    pub srl: Option<Vec<String>>,
    pub dep: Option<Vec<String>>,
    pub sdp: Option<Vec<String>>,
}

impl Vocab {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Vocab> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let vocab = serde_json::from_reader(reader)?;
        Ok(vocab)
    }
}

#[cfg(test)]
mod tests {
    use crate::vocabs::Vocab;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn test_vocab() {
        let file = File::open("../ltp.onnx/vocabs.json").unwrap();
        let reader = BufReader::new(file);

        let vocab: Vocab = serde_json::from_reader(reader).unwrap();

        println!("{:#?}", vocab);
    }
}
