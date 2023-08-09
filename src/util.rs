use crate::error::ParseError;
use image::flat::SampleLayout;
use image::{GrayImage, ImageBuffer, Pixel};
use ndarray::{Array3, Dim, Ix, ShapeBuilder, StrideShape};

pub fn image_to_array3<P: Pixel>(
    image: ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<Array3<P::Subpixel>, ParseError> {
    let array_shape = array3_shape_from_image_layout(image.sample_layout());
    let image_data = image.into_vec();
    let array = Array3::from_shape_vec(array_shape, image_data)?;
    Ok(array)
}

pub fn array3_shape_from_image_layout(layout: SampleLayout) -> StrideShape<Dim<[Ix; 3]>> {
    let SampleLayout {
        channels,
        channel_stride,
        height,
        height_stride,
        width,
        width_stride,
    } = layout;

    let shape = (height as usize, width as usize, channels as usize);
    let strides = (height_stride, width_stride, channel_stride);
    shape.strides(strides)
}
