mod error;
mod layout;
mod util;

use crate::layout::ROOM_TYPES;
use crate::util::{array3_shape_from_image_layout, image_to_array3};
use image::RgbImage;
use ndarray::{
    array, azip, s, Array, Array1, Array2, Array3, ArrayBase, ArrayView, Axis, Dimension,
    IntoNdProducer, NdProducer, OwnedRepr, RawData, ViewRepr, Zip,
};
use ndarray_stats::QuantileExt;
use polyfit_rs::polyfit_rs::polyfit;
use std::ops::Deref;
use std::path::PathBuf;

fn get_lsun_res() {
    let i = 1;

    let im_h = 512usize;
    let im_w = 512usize;

    let im_res = (im_w, im_h);

    let results_dir =
        PathBuf::from("/Users/richardkuodis/development/pytorch-layoutnet/res/lsun_tr_gt");

    let im = image::open(results_dir.join("img").join(format!("{i}.png")))
        .unwrap()
        .to_rgb8();

    let edg = image::open(results_dir.join("edg").join(format!("{i}.png")))
        .unwrap()
        .to_rgb8();

    // TODO: convert RgbImage to Array3

    assert_eq!(im.width(), im_w as u32);
    assert_eq!(im.height(), im_h as u32);
    assert_eq!(edg.width(), im_w as u32);
    assert_eq!(edg.height(), im_h as u32);

    let samples = edg.as_flat_samples();
    println!("Layoit: {:?}", samples.layout);
    // let im = Array3::from_shape_vec((im_h, im_w, 3), im.to_vec()).unwrap();
    // let edg = Array3::from_shape_vec((im_h, im_w, 3), edg.to_vec()).unwrap();
    //
    //
    // println!("{:?}", edg[(101, 20, 0)]);

    let edg = image_to_array3(edg).unwrap();
    println!("{:?}", edg[(101, 20, 0)]);

    let corn: Array3<f32> = ndarray_npy::read_npy(format!(
        "/Users/richardkuodis/development/Bath/LayoutNet/out/cor_mat_{i}.npy"
    ))
    .unwrap();
    let mut corn = corn.permuted_axes([1, 2, 0]); // CHW -> HWC
    println!("corn shape: {:?}", corn.shape());

    let corn_f: Array3<f32> = ndarray_npy::read_npy(format!(
        "/Users/richardkuodis/development/Bath/LayoutNet/out/cor_mat_flip_{i}.npy"
    ))
    .unwrap();
    let mut corn_f = corn_f.permuted_axes([1, 2, 0]); // CHW -> HWC
    println!("corn_f shape: {:?}", corn_f.shape());

    let r_t: Array2<f32> = ndarray_npy::read_npy(format!(
        "/Users/richardkuodis/development/Bath/LayoutNet/out/type_{i}.npy"
    ))
    .unwrap();
    let r_t = r_t.mean_axis(Axis(0)).unwrap();
    let record_id = r_t.argmax().unwrap() as usize;
    println!("r_t: {r_t:?}");
    println!("record_id: {record_id:?}");

    let room_t = &(*ROOM_TYPES)[record_id];
    println!("room_t: {room_t:?}");

    match room_t.typeid {
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
        _ => {
            panic!("Room type {} not supported yet!", room_t.typeid);
        }
    }

    // Find corner
    let mut point: Vec<Vec<usize>> = vec![];
    for corner_map in room_t.cornermap.iter() {
        let corn_idx = (corner_map - 1) as usize;

        let mut mp: Array2<f32> =
            &corn.slice(s![.., .., corn_idx]) + &corn_f.slice(s![.., .., corn_idx]);
        mp.slice_mut(s![.., 0]).fill(0.0);
        mp.slice_mut(s![.., im_w - 1]).fill(0.0);
        mp.slice_mut(s![0, ..]).fill(0.0);
        mp.slice_mut(s![im_h - 1, ..]).fill(0.0);

        let mut mp_msk = Array::from_elem((im_h, im_w), 0.);

        let threshold = 255.0 * 0.1;
        match room_t.typeid {
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

                        let (_pt_y, pt_x) = mp_t.argmax().unwrap();

                        mp_msk
                            .slice_mut(s![.., ..(usize::max(pt_x - 50, 1) + 1)])
                            .fill(0.);
                        mp_msk
                            .slice_mut(s![.., usize::min(pt_x + 50, im_w - 1)..])
                            .fill(0.);

                        // let flat = mp_t.iter().flatten();

                        // todo!();
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
                        panic!(
                            "Unexpected corner map {} for room type {} not supported yet!",
                            corner_map,
                            room_t.typeid
                        );
                    },
                }
            }
            _ => {
                panic!("Room type {} not supported yet!", room_t.typeid);
            }
        }

        mp *= &mp_msk;

        let (pt_x, pt_y) = (mp / 2.).argmax().unwrap();
        point.push(vec![pt_x, pt_y]);
    }

    let point = Array2::from_shape_vec((point.len(), 2), point.iter().flatten().collect()).unwrap();
    println!("point: {point:?}");

    let point_res = point.slice(s![.., ..;-1]);
    println!("point_res: {point_res:?}");

    // We skip the cor_res scaling, assume that input image is of width 512. Rescaling will be handled later.
    let point_ref_res = point_res.to_owned();
    println!("point_ref_res: {point_ref_res:?}");
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

fn ravel_fortran_order<T>(array: &Array2<T>) -> Array1<T>
where
    T: Clone,
{
    array.t().iter().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use matfile::{MatFile, NumericData};
    use ndarray::{Array1, Axis, ShapeBuilder, Zip};
    use ndarray_stats::QuantileExt;
    use std::any::Any;
    use std::fs::File;
    use std::path::PathBuf;

    #[test]
    fn get_lsun_res_works() {
        get_lsun_res();
    }

    #[test]
    fn polyfit_works() {
        let x = [1.0f32, 2.0f32];
        let y = [5.0f32, 4.0f32];

        // Note that output is reversed compared to numpy:
        // Vector index indicates the degree, thus:
        // Power of x^0 = degrees[0];
        // Power of x^1 = degrees[1];
        // Power of x^2 = degrees[2];
        // ...
        let degrees = polyfit(&x, &y, 1).unwrap();
        println!("{degrees:?}");

        // 6 - 1x
    }

    #[test]
    fn load_corn_matfile() {
        let path = "/Users/richardkuodis/development/Bath/LayoutNet/out/cor_mat.npy";
        let corn: Array3<f32> = ndarray_npy::read_npy(path).unwrap();
        let corn = corn.permuted_axes([1, 2, 0]);
        println!("{:?}", corn.shape());
    }

    #[test]
    fn matfile_works() {
        let path = "/Users/richardkuodis/development/pytorch-layoutnet/res/lsun_tr_gt/type/1.mat";
        let type_mat_file = File::open(PathBuf::from(path)).unwrap();
        let type_mat = MatFile::parse(type_mat_file).unwrap();

        println!("{type_mat:?}");
        let type_array = type_mat.find_by_name("x").unwrap();
        println!("{type_array:?}");
        let shape = type_array.size();
        println!("Shape: {shape:?}");

        let data = match type_array.data().clone() {
            NumericData::Double { real, imag } => real,
            _ => panic!(),
        };
        println!("{data:?}");

        let data_arr = ndarray::Array2::from_shape_vec(
            (shape[0], shape[1]).strides((1, 2)), // Stride 1 to pass to along axis[0], and 2 along axis[1]
            data,
        )
        .unwrap();
        println!("Array: {data_arr:?}");
        let data_arr = data_arr.mean_axis(Axis(0)).unwrap();
        println!("Mean: {data_arr:?}");
        let room_type = data_arr.argmax().unwrap() as usize;
        println!("{room_type:?}");
    }

    #[test]
    fn maximum_works() {
        let array1 = Array1::from_vec(vec![2, 3, 4]);
        let array2 = Array1::from_vec(vec![1, 5, 2]);

        let result = maximum(&array1, &array2);
        let result = maximum(&array1, &array2);

        let expected = Array1::from_vec(vec![2, 5, 4]);
        assert_eq!(expected, result);
    }

    #[test]
    fn broadcast_experiment() {
        let mut arr = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        println!("Before:\n{arr:?}");

        arr.slice_mut(s![.., 0]).fill(0);

        println!("After:\n{arr:?}");
    }

    #[test]
    fn fliplr_works() {
        let mut array = array![
            [[1, 10], [2, 20], [3, 30]],
            [[4, 40], [5, 50], [6, 60]],
            [[7, 70], [8, 80], [9, 90]]
        ];

        println!("Array: {array:?}");

        let flipped_array = array.slice(s![.., ..;-1, 0]);
        println!("Flipped: {flipped_array:?}");
    }

    #[test]
    fn ravel_works() {
        let arr = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        println!("arr: {arr:?}");

        let flat = arr.iter().cloned().collect::<Array1<_>>();
        println!("flat: {flat:?}");

        let flat_f = arr.t().iter().cloned().collect::<Array1<_>>();
        // let flat_f = arr.axis_iter(Axis(1)).into_iter().flatten().collect::<Array1<_>>();
        println!("flat_f: {flat_f:?}");
    }

    #[test]
    fn array_from_vec_of_tuples() {
        let data = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
        ];


        let arr = Array2::from_shape_vec((3, 2), data.iter().flatten().collect()).unwrap();
        println!("{arr:?}");
    }
}
