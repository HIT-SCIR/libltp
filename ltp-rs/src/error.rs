use thiserror::Error;
use onnxruntime::ndarray;

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
    ShapeError(ndarray::ShapeError),

    #[cfg(feature = "export")]
    #[error("{0}")]
    ArrowError(arrow::error::ArrowError),
}

impl From<onnxruntime::OrtError> for LTPError {
    fn from(status: onnxruntime::OrtError) -> Self {
        LTPError::OrtError(status)
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

#[cfg(feature = "export")]
impl From<arrow::error::ArrowError> for LTPError {
    fn from(status: arrow::error::ArrowError) -> Self {
        LTPError::ArrowError(status)
    }
}