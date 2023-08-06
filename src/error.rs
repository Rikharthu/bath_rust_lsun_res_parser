use ndarray::ShapeError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Wrong array shape: {0:?}")]
    ArrayShapeError(#[from] ShapeError),
}
