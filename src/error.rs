use ndarray::ShapeError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Wrong array shape: {0:?}")]
    ArrayShapeError(#[from] ShapeError),
    #[error("Unsupported room type: {0:?}")]
    UnsupportedRoomType(u8),
    #[error("Unsupported corner map {corner_map} for room type {room_type}")]
    UnexpectedCornerMapForRoomType { corner_map: u8, room_type: u8 },
}
