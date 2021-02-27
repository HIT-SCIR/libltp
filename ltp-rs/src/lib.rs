pub mod preinclude;

mod error;

mod eisner;
mod entities;
mod viterbi;

mod vocabs;
mod tokenizer;

mod interface;

pub use error::Result;
pub use error::LTPError;
pub use interface::{LTP, LTPResult, DEP, SDP};
