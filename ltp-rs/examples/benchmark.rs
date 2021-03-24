use clap::Clap;
use indicatif::ProgressBar;
use ltp_rs::preinclude::onnxruntime::GraphOptimizationLevel;
use ltp_rs::preinclude::serde_json;
use ltp_rs::{LTPError, LTP};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::Instant;

#[derive(Clap)]
#[clap(version = "1.0", author = "Feng Yunlong <ylfeng@ir.hit.edu.cn>")]
struct Opts {
    #[clap(short, long)]
    model: String,
    #[clap(short, long)]
    file: Option<String>,
    #[clap(short, long, default_value = "8")]
    batch_size: usize,
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

// todo ensure output format

fn main() -> Result<(), LTPError> {
    let opts: Opts = Opts::parse();

    let model_path = opts.model;
    #[cfg(feature = "cuda")]
    let mut ltp = LTP::new_with_cuda_options(&model_path, GraphOptimizationLevel::All, 16, 0)?;

    #[cfg(not(feature = "cuda"))]
    let mut ltp = LTP::new_with_options(&model_path, GraphOptimizationLevel::All, 16)?;
    let batch_size = opts.batch_size;

    let start = Instant::now();
    match opts.file {
        Some(path) => {
            // File hosts must exist in current path before this produces output
            if let Ok(lines) = read_lines(path) {
                // Consumes the iterator, returns an (Optional) String
                let bar = ProgressBar::new(10000);
                let mut batch = vec![];
                for (idx, line) in lines.enumerate() {
                    if let Ok(line) = line {
                        batch.push(line)
                    }
                    if (idx + 1) % batch_size == 0 {
                        // todo output
                        ltp.pipeline_batch(&batch)?;
                        bar.inc(batch_size as u64);
                        batch.clear();
                    }
                }
                bar.finish();
            }
        }
        None => {
            let sentence1 = String::from("我爱赛尔!");
            let sentence2 = String::from("他叫汤姆去拿外衣。");
            let sentence3 = String::from("同时发表一组阐述这次会议主要精神的评论员文章。");

            let sentences = vec![sentence1, sentence2, sentence3];
            let result = ltp.pipeline_batch(&sentences)?;

            let j = serde_json::to_string_pretty(&result).unwrap();
            println!("Batch=3: {}", j);
        }
    }
    let duration = start.elapsed();
    println!(
        "Done! Cost: {}s",
        duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9
    );
    Ok(())
}
