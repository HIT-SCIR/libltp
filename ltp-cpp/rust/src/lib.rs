use cxx::CxxString;
use ltp_rs::{LTP, LTPResult, Result};

pub struct Interface(LTP);

pub struct InterfaceResult(LTPResult);


#[cxx::bridge]
mod ffi {
    extern "Rust" {
        type Interface;
        type InterfaceResult;

        fn ltp_init(path: &CxxString) -> Result<Box<Interface>>;
        fn pipeline(self: &mut Interface, sentences: &Vec<String>) -> Vec<InterfaceResult>;

        fn len(self: &InterfaceResult) -> usize;

        fn seg(self: &InterfaceResult) -> &Vec<String>;
        fn pos(self: &InterfaceResult) -> &Vec<String>;
        fn ner(self: &InterfaceResult) -> &Vec<String>;
        fn srl(self: &InterfaceResult, idx: usize) -> &Vec<String>;
        fn dep_arc(self: &InterfaceResult, idx: usize) -> usize;
        fn dep_rel(self: &InterfaceResult, idx: usize) -> &String;

        fn sdp_len(self: &InterfaceResult) -> usize;
        fn sdp_src(self: &InterfaceResult, idx: usize) -> usize;
        fn sdp_tgt(self: &InterfaceResult, idx: usize) -> usize;
        fn sdp_rel(self: &InterfaceResult, idx: usize) -> &String;
    }
}

fn ltp_init(path: &CxxString) -> Result<Box<Interface>> {
    return Ok(Box::new(Interface(LTP::new(&path.to_string())?)));
}

impl Interface {
    pub fn pipeline(&mut self, sentences: &Vec<String>) -> Vec<InterfaceResult> {
        match self.0.pipeline_batch(sentences) {
            Ok(results) => {
                let mut res: Vec<InterfaceResult> = vec![];
                for result in results {
                    res.push(InterfaceResult(result));
                }
                res
            }
            Err(_e) => {
                vec![]
            }
        }
    }
}

impl InterfaceResult {
    pub fn len(&self) -> usize { self.0.seg.as_ref().unwrap().len() }
    pub fn seg(&self) -> &Vec<String> { self.0.seg.as_ref().unwrap() }
    pub fn pos(&self) -> &Vec<String> { self.0.pos.as_ref().unwrap() }
    pub fn ner(&self) -> &Vec<String> { self.0.ner.as_ref().unwrap() }
    pub fn srl(&self, idx: usize) -> &Vec<String> {
        self.0.srl.as_ref().unwrap()[idx].as_ref()
    }
    pub fn dep_arc(&self, idx: usize) -> usize { self.0.dep.as_ref().unwrap()[idx].arc }
    pub fn dep_rel(&self, idx: usize) -> &String { &self.0.dep.as_ref().unwrap()[idx].rel }

    pub fn sdp_len(&self) -> usize { self.0.seg.as_ref().unwrap().len() }
    pub fn sdp_src(&self, idx: usize) -> usize { self.0.sdp.as_ref().unwrap()[idx].src }
    pub fn sdp_tgt(&self, idx: usize) -> usize { self.0.sdp.as_ref().unwrap()[idx].tgt }
    pub fn sdp_rel(&self, idx: usize) -> &String { &self.0.sdp.as_ref().unwrap()[idx].rel }
}

#[cfg(test)]
mod tests {
    use cxx::let_cxx_string;
    use crate::ltp_init;

    #[test]
    fn test_interface() {
        let_cxx_string!(path = "onnx-small");
        let interface = ltp_init(&*path);
    }
}
