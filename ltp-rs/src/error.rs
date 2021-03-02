use crate::preinclude::onnxruntime::ndarray;
use crate::preinclude::onnxruntime::tensor::ort_owned_tensor::TensorExtractError;
use thiserror::Error;

/// Type alias for the `Result`
pub type Result<T> = std::result::Result<T, LTPError>;

/// Error type centralizing all possible errors
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum LTPError {
    #[error("{0}")]
    IOError(std::io::Error),

    #[error("{0}")]
    SerdeError(serde_json::Error),

    #[error("{0}")]
    OrtError(onnxruntime::OrtError),

    #[error("{0}")]
    TensorExtractError(TensorExtractError),

    #[error("{0}")]
    ShapeError(ndarray::ShapeError),
}

impl From<onnxruntime::OrtError> for LTPError {
    fn from(status: onnxruntime::OrtError) -> Self {
        LTPError::OrtError(status)
    }
}

impl From<TensorExtractError> for LTPError {
    fn from(status: TensorExtractError) -> Self {
        LTPError::TensorExtractError(status)
    }
}

impl From<ndarray::ShapeError> for LTPError {
    fn from(status: ndarray::ShapeError) -> Self {
        LTPError::ShapeError(status)
    }
}

impl From<serde_json::Error> for LTPError {
    fn from(status: serde_json::Error) -> Self {
        LTPError::SerdeError(status)
    }
}

impl From<std::io::Error> for LTPError {
    fn from(status: std::io::Error) -> Self {
        LTPError::IOError(status)
    }
}
