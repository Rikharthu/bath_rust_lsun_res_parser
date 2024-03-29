mod error;
mod layout;

use crate::error::ParseError;
use crate::layout::{LayoutData, ROOM_TYPES};
use ndarray::{
    array, concatenate, s, Array, Array1, Array2, Array3, ArrayBase, ArrayView2, Axis, Dimension,
    NewAxis, OwnedRepr, Zip,
};
use ndarray_stats::QuantileExt;
use polyfit_rs::polyfit_rs::polyfit;
use std::ops::Deref;

pub type Point = (i32, i32);

pub type Line = (Point, Point);

#[derive(Clone, Debug)]
pub struct RoomLayoutInfo {
    // TODO: better use enum
    pub room_type: u8,
    pub lines: Vec<Line>,
}

pub fn parse_lsun_results(
    edges: Array3<f32>,
    corners: Array3<f32>,
    corners_flip: Array3<f32>,
    room_type: Array2<f32>,
) -> Result<RoomLayoutInfo, ParseError> {
    let edges_shape = edges.shape();
    // CHW
    let im_h = edges_shape[1];
    let im_w = edges_shape[2];
    let im_res = (im_w, im_h);

    // Scale to u8 since original code assumes that edges are saved as RGB images
    let edg = edges.mapv(|x| (x * 255.) as u8).permuted_axes([1, 2, 0]);
    // CHW -> HWC
    let mut corn = corners.permuted_axes([1, 2, 0]);
    let mut corn_f = corners_flip.permuted_axes([1, 2, 0]);
    let r_t = room_type;

    let r_t = r_t.mean_axis(Axis(0)).unwrap();
    let record_id = r_t.argmax().unwrap() as usize;

    // FIXME: we parse room type 7 as room type 1
    let room_t = &(*ROOM_TYPES)[record_id];

    match room_t.typeid {
        0 => {
            let corn_t = corn_f.slice(s![.., .., 0]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 6]).to_owned();
            corn_f.slice_mut(s![.., .., 0]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 6]).assign(&corn_t);

            let corn_t = corn_f.slice(s![.., .., 2]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 4]).to_owned();
            corn_f.slice_mut(s![.., .., 2]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 4]).assign(&corn_t);

            let corn_t = corn_f.slice(s![.., .., 1]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 7]).to_owned();
            corn_f.slice_mut(s![.., .., 1]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 7]).assign(&corn_t);

            let corn_t = corn_f.slice(s![.., .., 3]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 5]).to_owned();
            corn_f.slice_mut(s![.., .., 3]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 5]).assign(&corn_t);

            let zeros = Array::zeros((im_w, im_h));

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 2]).to_owned() - corn.slice(s![.., .., 3]).to_owned()),
            );
            corn.slice_mut(s![.., .., 2]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 4]).to_owned() - corn.slice(s![.., .., 5]).to_owned()),
            );
            corn.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 0]).to_owned() - corn.slice(s![.., .., 1]).to_owned()),
            );
            corn.slice_mut(s![.., .., 0]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 6]).to_owned() - corn.slice(s![.., .., 7]).to_owned()),
            );
            corn.slice_mut(s![.., .., 6]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 2]).to_owned() - corn_f.slice(s![.., .., 3]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 2]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 4]).to_owned() - corn_f.slice(s![.., .., 5]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 0]).to_owned() - corn_f.slice(s![.., .., 1]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 0]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 6]).to_owned() - corn_f.slice(s![.., .., 7]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 6]).assign(&max_res);

            corn.slice_mut(s![.., 399, 0]).fill(0.);
            corn.slice_mut(s![.., 399.., 2]).fill(0.);
            corn.slice_mut(s![.., 0..112, 6]).fill(0.);

            let a = corn.slice(s![.., .., 4]).to_owned();
            let b = corn.slice(s![.., .., 2]).to_owned();
            let max_res = maximum::<f32, _>(&(&corn.slice(s![.., .., 4]) - &b), &zeros);
            corn.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(&(&corn.slice(s![.., .., 2]) - &a), &zeros);
            corn.slice_mut(s![.., .., 2]).assign(&max_res);

            let a = corn.slice(s![.., .., 0]).to_owned();
            let b = corn.slice(s![.., .., 6]).to_owned();
            let max_res = maximum::<f32, _>(&(&corn.slice(s![.., .., 0]) - &b), &zeros);
            corn.slice_mut(s![.., .., 0]).assign(&max_res);
            let max_res = maximum::<f32, _>(&(&corn.slice(s![.., .., 6]) - &a), &zeros);
            corn.slice_mut(s![.., .., 6]).assign(&max_res);

            let a = corn_f.slice(s![.., .., 4]).to_owned();
            let b = corn_f.slice(s![.., .., 2]).to_owned();
            let max_res = maximum::<f32, _>(&(&corn_f.slice(s![.., .., 4]) - &b), &zeros);
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(&(&corn_f.slice(s![.., .., 2]) - &a), &zeros);
            corn_f.slice_mut(s![.., .., 2]).assign(&max_res);

            let a = corn_f.slice(s![.., .., 0]).to_owned();
            let b = corn_f.slice(s![.., .., 6]).to_owned();
            let max_res = maximum::<f32, _>(&(&corn_f.slice(s![.., .., 0]) - &b), &zeros);
            corn_f.slice_mut(s![.., .., 0]).assign(&max_res);
            let max_res = maximum::<f32, _>(&(&corn_f.slice(s![.., .., 6]) - &a), &zeros);
            corn_f.slice_mut(s![.., .., 6]).assign(&max_res);
        }
        // FIXME: Since room type 7 is not implemented in Matlab, will parse it as room type 1
        1 | 7 => {
            let corn_t = corn_f.slice(s![.., .., 0]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 6]).to_owned();
            corn_f.slice_mut(s![.., .., 0]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 6]).assign(&corn_t);

            let corn_t = corn_f.slice(s![.., .., 2]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 4]).to_owned();
            corn_f.slice_mut(s![.., .., 2]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 4]).assign(&corn_t);

            let corn_t = corn_f.slice(s![.., .., 3]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 5]).to_owned();
            corn_f.slice_mut(s![.., .., 3]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 5]).assign(&corn_t);

            let zeros = Array::zeros((im_w, im_h));

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 2]).to_owned() - corn.slice(s![.., .., 3]).to_owned()),
            );
            corn.slice_mut(s![.., .., 2]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 4]).to_owned() - corn.slice(s![.., .., 5]).to_owned()),
            );
            corn.slice_mut(s![.., .., 4]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 2]).to_owned() - corn_f.slice(s![.., .., 3]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 2]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 4]).to_owned() - corn_f.slice(s![.., .., 5]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);

            let a = corn.slice(s![.., .., 4]).to_owned();
            let b = corn.slice(s![.., .., 2]).to_owned();
            let max_res = maximum::<f32, _>(&(&corn.slice(s![.., .., 4]) - &b), &zeros);
            corn.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(&(&corn.slice(s![.., .., 2]) - &a), &zeros);
            corn.slice_mut(s![.., .., 2]).assign(&max_res);

            let a = corn.slice(s![.., .., 0]).to_owned();
            let b = corn.slice(s![.., .., 6]).to_owned();
            let max_res = maximum::<f32, _>(&(&corn.slice(s![.., .., 0]) - &b), &zeros);
            corn.slice_mut(s![.., .., 0]).assign(&max_res);
            let max_res = maximum::<f32, _>(&(&corn.slice(s![.., .., 6]) - &a), &zeros);
            corn.slice_mut(s![.., .., 6]).assign(&max_res);

            let a = corn_f.slice(s![.., .., 4]).to_owned();
            let b = corn_f.slice(s![.., .., 2]).to_owned();
            let max_res = maximum::<f32, _>(&(&corn_f.slice(s![.., .., 4]) - &b), &zeros);
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(&(&corn_f.slice(s![.., .., 2]) - &a), &zeros);
            corn_f.slice_mut(s![.., .., 2]).assign(&max_res);

            let a = corn_f.slice(s![.., .., 0]).to_owned();
            let b = corn_f.slice(s![.., .., 6]).to_owned();
            let max_res = maximum::<f32, _>(&(&corn_f.slice(s![.., .., 0]) - &b), &zeros);
            corn_f.slice_mut(s![.., .., 0]).assign(&max_res);
            let max_res = maximum::<f32, _>(&(&corn_f.slice(s![.., .., 6]) - &a), &zeros);
            corn_f.slice_mut(s![.., .., 6]).assign(&max_res);
        }
        2 => {
            let corn_t = corn_f.slice(s![.., .., 1]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 7]).to_owned();
            corn_f.slice_mut(s![.., .., 1]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 7]).assign(&corn_t);

            let corn_t = corn_f.slice(s![.., .., 0]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 6]).to_owned();
            corn_f.slice_mut(s![.., .., 0]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 6]).assign(&corn_t);

            let corn_t = corn_f.slice(s![.., .., 2]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 4]).to_owned();
            corn_f.slice_mut(s![.., .., 2]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 4]).assign(&corn_t);

            let zeros = Array::zeros((im_w, im_h));

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 0]).to_owned() - corn.slice(s![.., .., 1]).to_owned()),
            );
            corn.slice_mut(s![.., .., 0]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 6]).to_owned() - corn.slice(s![.., .., 7]).to_owned()),
            );
            corn.slice_mut(s![.., .., 6]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 0]).to_owned() - corn_f.slice(s![.., .., 1]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 0]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 6]).to_owned() - corn_f.slice(s![.., .., 7]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 6]).assign(&max_res);
        }
        3 => {
            let corn_t = corn_f.slice(s![.., .., 0]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 7]).to_owned();
            corn_f.slice_mut(s![.., .., 0]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 7]).assign(&corn_t);

            let zeros = Array::zeros((im_w, im_h));

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 6]).to_owned() - corn.slice(s![.., .., 0]).to_owned()),
            );
            corn.slice_mut(s![.., .., 6]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 6]).to_owned() - corn_f.slice(s![.., .., 0]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 6]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 6]).to_owned() - corn.slice(s![.., .., 7]).to_owned()),
            );
            corn.slice_mut(s![.., .., 6]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 6]).to_owned() - corn_f.slice(s![.., .., 7]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 6]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 4]).to_owned() - corn.slice(s![.., .., 2]).to_owned()),
            );
            corn.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 4]).to_owned() - corn_f.slice(s![.., .., 2]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);
        }
        4 => {
            let corn_t = corn_f.slice(s![.., .., 2]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 5]).to_owned();
            corn_f.slice_mut(s![.., .., 2]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 5]).assign(&corn_t);

            let zeros = Array::zeros((im_w, im_h));

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 4]).to_owned() - corn.slice(s![.., .., 2]).to_owned()),
            );
            corn.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 4]).to_owned() - corn_f.slice(s![.., .., 2]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 4]).to_owned() - corn.slice(s![.., .., 5]).to_owned()),
            );
            corn.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 4]).to_owned() - corn_f.slice(s![.., .., 5]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 6]).to_owned() - corn.slice(s![.., .., 0]).to_owned()),
            );
            corn.slice_mut(s![.., .., 6]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 6]).to_owned() - corn_f.slice(s![.., .., 0]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 6]).assign(&max_res);
        }
        5 => {
            let corn_t = corn_f.slice(s![.., .., 0]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 7]).to_owned();
            corn_f.slice_mut(s![.., .., 0]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 7]).assign(&corn_t);

            let corn_t = corn_f.slice(s![.., .., 2]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 5]).to_owned();
            corn_f.slice_mut(s![.., .., 2]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 5]).assign(&corn_t);

            let zeros = Array::zeros((im_w, im_h));

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 6]).to_owned() - corn.slice(s![.., .., 0]).to_owned()),
            );
            corn.slice_mut(s![.., .., 6]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 6]).to_owned() - corn_f.slice(s![.., .., 0]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 6]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 4]).to_owned() - corn.slice(s![.., .., 2]).to_owned()),
            );
            corn.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 4]).to_owned() - corn_f.slice(s![.., .., 2]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 6]).to_owned() - corn.slice(s![.., .., 7]).to_owned()),
            );
            corn.slice_mut(s![.., .., 6]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 6]).to_owned() - corn_f.slice(s![.., .., 7]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 6]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 4]).to_owned() - corn.slice(s![.., .., 5]).to_owned()),
            );
            corn.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 4]).to_owned() - corn_f.slice(s![.., .., 5]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);
        }
        6 => {
            let corn_t = corn_f.slice(s![.., .., 0]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 6]).to_owned();
            corn_f.slice_mut(s![.., .., 0]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 6]).assign(&corn_t);

            let corn_t = corn_f.slice(s![.., .., 2]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 4]).to_owned();
            corn_f.slice_mut(s![.., .., 2]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 4]).assign(&corn_t);

            let zeros = Array::zeros((im_w, im_h));

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 2]).to_owned() - corn.slice(s![.., .., 3]).to_owned()),
            );
            corn.slice_mut(s![.., .., 2]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 2]).to_owned() - corn_f.slice(s![.., .., 3]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 2]).assign(&max_res);

            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn.slice(s![.., .., 4]).to_owned() - corn.slice(s![.., .., 5]).to_owned()),
            );
            corn.slice_mut(s![.., .., 4]).assign(&max_res);
            let max_res = maximum::<f32, _>(
                &zeros,
                &(corn_f.slice(s![.., .., 4]).to_owned() - corn_f.slice(s![.., .., 5]).to_owned()),
            );
            corn_f.slice_mut(s![.., .., 4]).assign(&max_res);
        }
        // 7 => (), // Not implemented in Matlab code
        8 => {
            let corn_t = corn_f.slice(s![.., .., 0]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 6]).to_owned();
            corn_f.slice_mut(s![.., .., 0]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 6]).assign(&corn_t);
        } // FIXME: Not implemented in Matlab code, tried re-implemented on my own
        9 => {
            let corn_t = corn_f.slice(s![.., .., 2]).to_owned();
            let corn_b = corn_f.slice(s![.., .., 4]).to_owned();
            corn_f.slice_mut(s![.., .., 2]).assign(&corn_b);
            corn_f.slice_mut(s![.., .., 4]).assign(&corn_t);
        }
        10 => (), // Do nothing
        _ => {
            return Err(ParseError::UnsupportedRoomType(room_t.typeid));
        }
    }

    // Find corner
    let mut point: Vec<Vec<usize>> = vec![];
    for corner_map in room_t.cornermap.iter() {
        let corn_idx = (corner_map - 1) as usize;

        let mut mp: Array2<f32> =
            &corn.slice(s![.., .., corn_idx]) + &corn_f.slice(s![.., ..;-1, corn_idx]);
        mp.slice_mut(s![.., 0]).fill(0.0);
        mp.slice_mut(s![.., im_w - 1]).fill(0.0);
        mp.slice_mut(s![0, ..]).fill(0.0);
        mp.slice_mut(s![im_h - 1, ..]).fill(0.0);

        let mut mp_msk = Array::from_elem((im_h, im_w), 0.);

        let threshold = 255.0 * 0.1;
        match room_t.typeid {
            0 => match corner_map {
                5 | 3 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                1 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);

                    let mut mp_t: Array2<f32> = (corn.slice(s![.., .., 2]).to_owned()
                        + &corn_f.slice(s![.., ..;-1, 2]))
                        / 2.;
                    pad_zeros(&mut mp_t);
                    mp_t *= &mp_msk;

                    let (_pt_y, pt_x) = mp_t.argmax().unwrap();

                    mp_msk
                        .slice_mut(s![.., ..((i64::max(pt_x as i64 - 50, 1) + 1) as usize)])
                        .fill(0.);
                    mp_msk
                        .slice_mut(s![
                            ..,
                            (i64::min(pt_x as i64 + 50, im_w as i64 - 1) as usize)..
                        ])
                        .fill(0.);
                }
                7 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);

                    let mut mp_t: Array2<f32> = (corn.slice(s![.., .., 4]).to_owned()
                        + &corn_f.slice(s![.., ..;-1, 4]))
                        / 2.;
                    pad_zeros(&mut mp_t);
                    mp_t *= &mp_msk;

                    let (_pt_y, pt_x) = mp_t.argmax().unwrap();

                    mp_msk
                        .slice_mut(s![.., ..((i64::max(pt_x as i64 - 50, 1) + 1) as usize)])
                        .fill(0.);
                    mp_msk
                        .slice_mut(s![
                            ..,
                            (i64::min(pt_x as i64 + 50, im_w as i64 - 1) as usize)..
                        ])
                        .fill(0.);
                }
                8 | 2 => {
                    mp_msk = edg
                        .slice(s![.., .., 0])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                4 | 6 => {
                    mp_msk = edg
                        .slice(s![.., .., 2])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                _ => {
                    return Err(ParseError::UnexpectedCornerMapForRoomType {
                        corner_map: *corner_map,
                        room_type: room_t.typeid,
                    });
                }
            },
            // FIXME: Since room type 7 is not implemented in Matlab, will parse it as room type 1
            1 | 7 => match corner_map {
                7 | 1 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                3 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);

                    let mut mp_t: Array2<f32> = (corn.slice(s![.., .., 0]).to_owned()
                        + &corn_f.slice(s![.., ..;-1, 0]))
                        / 2.;
                    pad_zeros(&mut mp_t);
                    mp_t *= &mp_msk;

                    let (_pt_y, pt_x) = mp_t.argmax().unwrap();

                    mp_msk
                        .slice_mut(s![
                            ..,
                            // Use max of pt_x vs.50 to prevent subtraction overflow
                            ..((i64::max(pt_x as i64 - 50, 1) + 1) as usize)
                        ])
                        .fill(0.);
                    mp_msk
                        .slice_mut(s![
                            ..,
                            (i64::min(pt_x as i64 + 50, im_w as i64 - 1) as usize)..
                        ])
                        .fill(0.);
                }
                5 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);

                    let mut mp_t: Array2<f32> = (corn.slice(s![.., .., 6]).to_owned()
                        + &corn_f.slice(s![.., ..;-1, 6]))
                        / 2.;
                    pad_zeros(&mut mp_t);
                    mp_t *= &mp_msk;

                    let (_pt_y, pt_x) = mp_t.argmax().unwrap();

                    mp_msk
                        .slice_mut(s![.., ..((i64::max(pt_x as i64 - 50, 1) + 1) as usize)])
                        .fill(0.);
                    mp_msk
                        .slice_mut(s![
                            ..,
                            (i64::min(pt_x as i64 + 50, im_w as i64 - 1) as usize)..
                        ])
                        .fill(0.);
                }
                4 | 6 => {
                    mp_msk = edg
                        .slice(s![.., .., 2])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                _ => {
                    return Err(ParseError::UnexpectedCornerMapForRoomType {
                        corner_map: *corner_map,
                        room_type: room_t.typeid,
                    });
                }
            },
            2 => match corner_map {
                2 | 8 => {
                    mp_msk = edg
                        .slice(s![.., .., 0])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                5 | 3 | 1 | 7 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                _ => {
                    return Err(ParseError::UnexpectedCornerMapForRoomType {
                        corner_map: *corner_map,
                        room_type: room_t.typeid,
                    });
                }
            },
            3 => match corner_map {
                1 | 8 => {
                    mp_msk = edg
                        .slice(s![.., .., 0])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                5 | 7 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                _ => {
                    return Err(ParseError::UnexpectedCornerMapForRoomType {
                        corner_map: *corner_map,
                        room_type: room_t.typeid,
                    });
                }
            },
            4 => match corner_map {
                5 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                7 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);

                    let mut mp_t: Array2<f32> = (corn.slice(s![.., .., 4]).to_owned()
                        + &corn_f.slice(s![.., ..;-1, 4]))
                        / 2.;
                    pad_zeros(&mut mp_t);
                    mp_t *= &mp_msk;

                    let (_pt_y, pt_x) = mp_t.argmax().unwrap();

                    mp_msk
                        .slice_mut(s![.., ..((i64::max(pt_x as i64 - 50, 1) + 1) as usize)])
                        .fill(0.);
                    mp_msk
                        .slice_mut(s![
                            ..,
                            (i64::min(pt_x as i64 + 50, im_w as i64 - 1) as usize)..
                        ])
                        .fill(0.);
                }
                3 | 6 => {
                    mp_msk = edg
                        .slice(s![.., .., 2])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                _ => {
                    return Err(ParseError::UnexpectedCornerMapForRoomType {
                        corner_map: *corner_map,
                        room_type: room_t.typeid,
                    });
                }
            },
            5 => {
                match corner_map {
                    7 => {
                        mp_msk = edg
                            .slice(s![.., .., 1])
                            .mapv(|x| (x as f32 > threshold) as u8 as f32);
                    }
                    5 => {
                        mp_msk = edg
                            .slice(s![.., .., 1])
                            .mapv(|x| (x as f32 > threshold) as u8 as f32);

                        // We perform the flip on 2nd axis (column) using negative step
                        // At the same time we select the necessary channel
                        // Basically it is the same as
                        // 1) selecting channel: corn_f.slice(s![.., .., 6])
                        // 2) then flipping it .slice(s![.., ..;-1])
                        let mut mp_t: Array2<f32> = (corn.slice(s![.., .., 6]).to_owned()
                            + &corn_f.slice(s![.., ..;-1, 6]))
                            / 2.;
                        pad_zeros(&mut mp_t);
                        mp_t *= &mp_msk;

                        // Skipped the ravel part, just computing the argmax
                        let (_pt_y, pt_x) = mp_t.argmax().unwrap();

                        mp_msk
                            .slice_mut(s![.., ..((i64::max(pt_x as i64 - 50, 1) + 1) as usize)])
                            .fill(0.);
                        mp_msk
                            .slice_mut(s![
                                ..,
                                (i64::min(pt_x as i64 + 50, im_w as i64 - 1) as usize)..
                            ])
                            .fill(0.);
                    }
                    1 | 8 => {
                        mp_msk = edg
                            .slice(s![.., .., 0])
                            .mapv(|x| (x as f32 > threshold) as u8 as f32);
                    }
                    3 | 6 => {
                        mp_msk = edg
                            .slice(s![.., .., 2])
                            .mapv(|x| (x as f32 > threshold) as u8 as f32);
                    }
                    _ => {
                        return Err(ParseError::UnexpectedCornerMapForRoomType {
                            corner_map: *corner_map,
                            room_type: room_t.typeid,
                        });
                    }
                }
            }
            6 => match corner_map {
                1 | 7 => {
                    mp_msk = edg
                        .slice(s![.., .., 0])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                3 | 5 => {
                    mp_msk = edg
                        .slice(s![.., .., 2])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                _ => {
                    return Err(ParseError::UnexpectedCornerMapForRoomType {
                        corner_map: *corner_map,
                        room_type: room_t.typeid,
                    });
                }
            },
            // 7 => (), // No corners
            8 => match corner_map {
                1 | 7 => {
                    mp_msk = edg
                        .slice(s![.., .., 0]) // Ceiling seems to be in R (first) channel
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                _ => {
                    return Err(ParseError::UnexpectedCornerMapForRoomType {
                        corner_map: *corner_map,
                        room_type: room_t.typeid,
                    });
                }
            }, // Not implemented in Matlab, tried on my own
            9 => match corner_map {
                5 | 3 => {
                    mp_msk = edg
                        .slice(s![.., .., 2])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                _ => {
                    return Err(ParseError::UnexpectedCornerMapForRoomType {
                        corner_map: *corner_map,
                        room_type: room_t.typeid,
                    });
                }
            },
            10 => match corner_map {
                5 | 7 => {
                    mp_msk = edg
                        .slice(s![.., .., 1])
                        .mapv(|x| (x as f32 > threshold) as u8 as f32);
                }
                _ => {
                    return Err(ParseError::UnexpectedCornerMapForRoomType {
                        corner_map: *corner_map,
                        room_type: room_t.typeid,
                    });
                }
            },
            _ => {
                return Err(ParseError::UnsupportedRoomType(room_t.typeid));
            }
        }

        mp *= &mp_msk;

        let (pt_x, pt_y) = (mp / 2.).argmax().unwrap();
        point.push(vec![pt_x, pt_y]);
    }

    let point =
        Array2::from_shape_vec((point.len(), 2), point.into_iter().flatten().collect()).unwrap();

    let point_res = point.slice(s![.., ..;-1]);

    // We skip the cor_res scaling, assume that input image is of width 512. Rescaling will be handled later.
    let mut point_ref_res = point_res.into_owned();
    // Coordinates exceeding height
    point_ref_res
        .slice_mut(s![.., 0])
        .mapv_inplace(|v| v.clamp(0, im_res.1));
    // Coordinates exceeding width
    point_ref_res
        .slice_mut(s![.., 1])
        .mapv_inplace(|v| v.clamp(0, im_res.0));

    // Refine point
    let mut point: Array2<usize> = point_ref_res;
    let mut point_ref: Array2<f32> = point.mapv(|x| x as f32).to_owned();

    let p: Array2<f32> = array![
        [0., 0.],
        [0., im_res.0 as f32 + 0.01],
        [im_res.1 as f32 + 0.01, im_res.0 as f32 + 0.01],
        [im_res.1 as f32 + 0.01, 0.]
    ];
    let p = p.t().as_standard_layout().into_owned();

    let p = concatenate![Axis(1), p, p.slice(s![.., 0, NewAxis])];

    match room_t.typeid {
        0 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![1, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(2, 0)] as f32, point[(3, 0)] as f32],
                &[point[(2, 1)] as f32, point[(3, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(2, 0)] as f32, point[(2, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![3, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(4, 0)] as f32, point[(5, 0)] as f32],
                &[point[(4, 1)] as f32, point[(5, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(4, 0)] as f32, point[(4, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![100000., 100000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![5, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(6, 0)] as f32, point[(7, 0)] as f32],
                &[point[(6, 1)] as f32, point[(7, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(6, 0)] as f32, point[(6, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![100000., 100000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![7, ..]).assign(&x.t());
        }
        // FIXME: Since room type 7 is not implemented in Matlab, will parse it as room type 1
        1 | 7 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![(-1. - line_1[1]) / line_1[0], -1.]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![1, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(2, 0)] as f32],
                &[point[(0, 1)] as f32, point[(2, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![2, ..]).assign(&x.t());

            // if point[(3, 0)] == point[(4, 0)] {
            //     // FIXME: in Matlab and Python we add 0.01, but here `point` is of type usize
            //     point[(4, 0)] = point[(4, 0)] + 1
            // }

            // let mut line_1 = polyfit(
            //     &[point[(3, 0)] as f32, point[(4, 0)] as f32],
            //     &[point[(3, 1)] as f32, point[(4, 1)] as f32],
            //     1,
            // )
            // .unwrap();
            // line_1.reverse();
            // let mut s1 = Array2::<f32>::zeros((2, 2));
            // s1.slice_mut(s![.., 0])
            //     .assign(&array![point[(3, 0)] as f32, point[(3, 1)] as f32]);
            // s1.slice_mut(s![.., 1])
            //     .assign(&array![(-1. - line_1[1]) / line_1[0], -1.]);
            // let x = seg2poly(&s1.view(), &p.view());
            // point_ref.slice_mut(s![4, ..]).assign(&x.t());
            //
            // let mut line_1 = polyfit(
            //     &[point[(3, 0)] as f32, point[(5, 0)] as f32],
            //     &[point[(3, 1)] as f32, point[(5, 1)] as f32],
            //     1,
            // )
            // .unwrap();
            // line_1.reverse();
            // let mut s1 = Array2::<f32>::zeros((2, 2));
            // s1.slice_mut(s![.., 0])
            //     .assign(&array![point[(3, 0)] as f32, point[(3, 1)] as f32]);
            // s1.slice_mut(s![.., 1])
            //     .assign(&array![100000., 100000. * line_1[0] + line_1[1]]);
            // let x = seg2poly(&s1.view(), &p.view());
            // point_ref.slice_mut(s![5, ..]).assign(&x.t());
        }
        2 => (),
        3 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![1, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(3, 0)] as f32],
                &[point[(0, 1)] as f32, point[(3, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![100000., 100000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![3, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(2, 0)] as f32],
                &[point[(0, 1)] as f32, point[(2, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![(10000. - line_1[1]) / line_1[0], 10000.]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![2, ..]).assign(&x.t());
        }
        4 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![1, ..]).assign(&x.t());

            if point[(0, 0)] == point[(2, 0)] {
                point[(0, 0)] = point[(0, 0)] + 1;
            }

            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(2, 0)] as f32],
                &[point[(0, 1)] as f32, point[(2, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![(-1. - line_1[1]) / line_1[0], -1.]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![2, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(3, 0)] as f32],
                &[point[(0, 1)] as f32, point[(3, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![10000., 10000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![3, ..]).assign(&x.t());
        }
        5 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            )
            .unwrap();
            // polyfit_rs order is reverse to that of Python's numpy
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![1, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(2, 0)] as f32],
                &[point[(0, 1)] as f32, point[(2, 1)] as f32],
                1,
            )
            .unwrap();
            // polyfit_rs order is reverse to that of Python's numpy
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![10000., 10000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![2, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(3, 0)] as f32, point[(4, 0)] as f32],
                &[point[(3, 1)] as f32, point[(4, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(3, 0)] as f32, point[(3, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![4, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(3, 0)] as f32, point[(5, 0)] as f32],
                &[point[(3, 1)] as f32, point[(5, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(3, 0)] as f32, point[(3, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![10000., 10000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![5, ..]).assign(&x.t());
        }
        6 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![0, ..]).assign(&x.t());

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(1, 0)] as f32, point[(1, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![10000., 10000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![1, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(2, 0)] as f32, point[(3, 0)] as f32],
                &[point[(2, 1)] as f32, point[(3, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(2, 0)] as f32, point[(2, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![2, ..]).assign(&x.t());

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(3, 0)] as f32, point[(3, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![10000., 10000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![3, ..]).assign(&x.t());
        }
        7 => (), // Not implemented in Matlab code
        8 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![0, ..]).assign(&x.t());

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(1, 0)] as f32, point[(1, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![10000., 10000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![1, ..]).assign(&x.t());
        } // Not implemented in the original Matlab code, tried implementing on my own
        9 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![0, ..]).assign(&x.t());

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(1, 0)] as f32, point[(1, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![10000., 10000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![1, ..]).assign(&x.t());
        }
        10 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            )
            .unwrap();
            line_1.reverse();

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![(1. - line_1[1]) / line_1[0], -1.]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![0, ..]).assign(&x.t());

            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0])
                .assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1])
                .assign(&array![(10000. - line_1[1]) / line_1[0], 10000.]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![1, ..]).assign(&x.t());
        }
        _ => {
            todo!("Unhandled room type {}", room_t.typeid);
        }
    }

    let data = LayoutData {
        type_: room_t.typeid,
        point: point_ref,
        resolution: im_res,
    };

    let lines = get_segmentation(data);

    // FIXME: We parse room type 7 as type 1, thus select only lines 1 and 3
    // if record_id == 7 {
    //     let lines = vec![lines[1], lines[3]];
    //     return Ok(lines);
    // }

    Ok(RoomLayoutInfo {
        room_type: room_t.typeid,
        lines,
    })
}

fn get_segmentation(data: LayoutData) -> Vec<Line> {
    if data.type_ == 11 {
        todo!("Return None");
    }

    let type_data = &ROOM_TYPES[data.type_ as usize];

    let point = data.point;
    let lines = type_data.lines.clone();

    let point = point.mapv(|x| x as i32);

    let line_indices = lines.clone().mapv_into(|x| x - 1);
    let line_point_indices = line_indices.slice(s![.., 0]).to_owned();
    let mut pt1s = point.select(Axis(0), line_point_indices.as_slice().unwrap());

    pt1s.mapv_inplace(|p| if p < 0 { 0 } else { p });
    // TODO: check if it is really x, not y
    let res_x = data.resolution.0 as i32;
    pt1s.slice_mut(s![.., 0]).mapv_inplace(|x| {
        // TODO: do we need to clamp to [0; max_size -1] or [1; max_size] as in Python?
        x.clamp(0, res_x - 1)
    });
    let res_y = data.resolution.1 as i32;
    pt1s.slice_mut(s![.., 1])
        .mapv_inplace(|y| y.clamp(0, res_y - 1));

    let line_point_indices = line_indices.slice(s![.., 1]).to_owned();
    let mut pt2s = point.select(Axis(0), line_point_indices.as_slice().unwrap());
    pt2s.slice_mut(s![.., 0])
        .mapv_inplace(|x| x.clamp(0, res_x - 1));
    pt2s.slice_mut(s![.., 1])
        .mapv_inplace(|y| y.clamp(0, res_y - 1));

    let num_lines = pt1s.shape()[0];
    let mut line_coords = vec![];
    for i in 0..num_lines {
        let p1 = (pt1s[(i, 0)], pt1s[(i, 1)]);
        let p2 = (pt2s[(i, 0)], pt2s[(i, 1)]);
        line_coords.push((p1, p2));
    }

    line_coords
}

fn pad_zeros(array: &mut Array2<f32>) {
    array.slice_mut(s![.., 0]).fill(0.);
    array.slice_mut(s![.., -1]).fill(0.);
    array.slice_mut(s![0, ..]).fill(0.);
    array.slice_mut(s![-1, ..]).fill(0.);
}

fn maximum<T, D>(
    array1: &ArrayBase<OwnedRepr<T>, D>,
    array2: &ArrayBase<OwnedRepr<T>, D>,
) -> Array<T, D>
where
    D: Dimension,
    T: PartialOrd + Copy,
    for<'a> &'a T: Deref<Target = T>,
{
    Zip::from(array1)
        .and(array2)
        .map_collect(|a, b| if a >= b { *a } else { *b })
}

/// Check if a line segment s intersects with a polygon P.
///
/// Parameters:
///     s: (2 x 2) array where
///         s[:, 0] is the first point
///         s[:, 1] is the second point of the segment
///     p: is (2 x n) array, each column is a vertice
///
/// Returns:
///     A (2 x m) array, each column is an intersection point
fn seg2poly(s1: &ArrayView2<f32>, p: &ArrayView2<f32>) -> Array1<f32> {
    let a = s1.slice(s![.., 0]).insert_axis(Axis(1));
    let m: Array2<f32> = p - &a;

    let b = &s1.slice(s![.., 1]).insert_axis(Axis(1)) - &a;
    let b: Array1<f32> = array![b[(0, 0)], b[(1, 0)]];

    let x: Array1<f32> = array![b[1], -b[0]].dot(&m);

    let sx = x.mapv(|x| x.signum());

    let ind: Array1<bool> = (&sx.slice(s![0..-1]) * &sx.slice(s![1..])).mapv(|x| x <= 0.);

    if ind.iter().any(|&x| x) {
        let ind: Vec<usize> = ind
            .indexed_iter()
            .filter(|(_index, &x)| x)
            .map(|(index, _)| index)
            .collect();

        // TODO: this doesn't actually match the original python code as there 2D array is created and selected
        let ind_t: Vec<usize> = ind.iter().map(|x| *x + 1).collect();
        let x1 = x.clone().select(Axis(0), &ind);
        let x2 = x.clone().select(Axis(0), &ind_t);

        let d: Array1<f32> = &b / (b[0].powi(2) + b[1].powi(2));

        let y1 = d.dot(&m.select(Axis(1), &ind));
        let y2 = d.dot(&m.select(Axis(1), &ind_t));

        let dx: Array1<f32> = &x2 - &x1;

        // We won't bother with the degenerate case of dx=0 and x1=0
        let y: Array1<f32> = (&y1 * &x2 - &y2 * &x1) / &dx;

        // Check if the cross point is inside the segment
        let ind: Array1<bool> = y.mapv(|y| y >= 0. && y < 1.);

        if ind.iter().any(|&x| x) {
            let where_ind: Vec<usize> = ind
                .indexed_iter()
                .filter(|(_index, &x)| x)
                .map(|(index, _)| index)
                .collect();
            // TODO: not sure if this `remove_axis(Axis(1))` acts the same as unsqueeze in all cases
            let x: Array1<f32> = &a.remove_axis(Axis(1)) + &b * &y.select(Axis(0), &where_ind);
            return x;
        } else {
            let x = Array1::<f32>::zeros((2,));
            return x;
        }
    } else {
        let x = Array1::<f32>::zeros((2,));
        return x;
    }
}
