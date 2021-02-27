use ltp_rs::preinclude::serde_json;
use ltp_rs::{LTP, LTPError};

// todo argparse (use clap)
// todo ensure input/output format

fn main() -> Result<(), LTPError> {
    let path = String::from("onnx-small");
    let mut ltp = LTP::new(&path)?;

    let sentence = String::from("他叫汤姆去拿外衣。");
    let result = ltp.pipeline(&sentence)?;

    let j = serde_json::to_string(&result).unwrap();
    println!("{}", j);

    let sentence2 = String::from("我爱赛尔!");
    let sentences = vec![sentence, sentence2];
    let result = ltp.pipeline_batch(&sentences)?;

    let j = serde_json::to_string_pretty(&result).unwrap();
    println!("{}", j);
    Ok(())
}
