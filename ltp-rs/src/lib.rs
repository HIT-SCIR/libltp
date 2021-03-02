pub mod preinclude;

mod error;

pub mod eisner;
pub mod entities;
pub mod viterbi;

mod tokenizer;
mod vocabs;

mod interface;

pub use error::LTPError;
pub use error::Result;
pub use interface::{LTPResult, DEP, LTP, SDP};
