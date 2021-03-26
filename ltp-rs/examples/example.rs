use ltp_rs::{LTPError, LTP};

fn main() -> Result<(), LTPError> {
    let mut ltp = LTP::new("path/to/model", 16)?;
    let sentences = vec![String::from("他叫汤姆去拿外衣。")];
    let result = ltp.pipeline_batch(&sentences)?;
    println!("{:?}", result);
    Ok(())
}
