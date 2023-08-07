mod error;
mod layout;
mod util;

use crate::layout::{LayoutData, ROOM_TYPES};
use crate::util::{array3_shape_from_image_layout, image_to_array3};
use image::RgbImage;
use ndarray::{array, azip, concatenate, s, stack, Array, Array1, Array2, Array3, ArrayBase, ArrayView, Axis, Dimension, IntoNdProducer, NdProducer, NewAxis, OwnedRepr, RawData, ViewRepr, Zip, ArrayView2};
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
                            corner_map, room_t.typeid
                        );
                    }
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

    let point =
        Array2::from_shape_vec((point.len(), 2), point.into_iter().flatten().collect()).unwrap();
    println!("point: {point:?}");

    let point_res = point.slice(s![.., ..;-1]);
    println!("point_res: {point_res:?}");

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
    println!("point_ref_res: {point_ref_res:?}");

    // Refine point
    let point: Array2<usize> = point_ref_res;
    let mut point_ref: Array2<f32> = point.mapv(|x| x as f32).to_owned();
    println!("point: {point:?}");

    let p: Array2<f32> = array![
        [0., 0.],
        [0., im_res.0 as f32 + 0.01],
        [im_res.1 as f32 + 0.01, im_res.0 as f32 + 0.01],
        [im_res.1 as f32 + 0.01, 0.]
    ];
    let mut p = p.t().as_standard_layout().into_owned();
    println!("p: {p:?}");

    let p = concatenate![Axis(1), p, p.slice(s![.., 0, NewAxis])];
    println!("p: {p:?}");

    match room_t.typeid {
        5 => {
            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(1, 0)] as f32],
                &[point[(0, 1)] as f32, point[(1, 1)] as f32],
                1,
            ).unwrap();
            // polyfit_rs order is reverse to that of Python's numpy
            line_1.reverse();
            println!("line_1: {line_1:?}");
            // TODO: Yes, this could be optimized on construction
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0]).assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1]).assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            println!("s1: {s1:?}");
            let x = seg2poly(&s1.view(), &p.view());
            println!("x: {x:?}");
            point_ref.slice_mut(s![1, ..]).assign(&x.t());
            println!("point_ref: {point_ref:?}");

            let mut line_1 = polyfit(
                &[point[(0, 0)] as f32, point[(2, 0)] as f32],
                &[point[(0, 1)] as f32, point[(2, 1)] as f32],
                1,
            ).unwrap();
            // polyfit_rs order is reverse to that of Python's numpy
            line_1.reverse();
            println!("line_1: {line_1:?}");
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0]).assign(&array![point[(0, 0)] as f32, point[(0, 1)] as f32]);
            s1.slice_mut(s![.., 1]).assign(&array![10000., 10000. * line_1[0] + line_1[1]]);
            println!("s1: {s1:?}");
            let x = seg2poly(&s1.view(), &p.view());
            println!("x: {x:?}");
            point_ref.slice_mut(s![2, ..]).assign(&x.t());
            println!("point_ref: {point_ref:?}");

            let mut line_1 = polyfit(
                &[point[(3, 0)] as f32, point[(4, 0)] as f32],
                &[point[(3, 1)] as f32, point[(4, 1)] as f32],
                1,
            ).unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0]).assign(&array![point[(3, 0)] as f32, point[(3, 1)] as f32]);
            s1.slice_mut(s![.., 1]).assign(&array![-100., -100. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![4, ..]).assign(&x.t());

            let mut line_1 = polyfit(
                &[point[(3, 0)] as f32, point[(5, 0)] as f32],
                &[point[(3, 1)] as f32, point[(5, 1)] as f32],
                1,
            ).unwrap();
            line_1.reverse();
            let mut s1 = Array2::<f32>::zeros((2, 2));
            s1.slice_mut(s![.., 0]).assign(&array![point[(3, 0)] as f32, point[(3, 1)] as f32]);
            s1.slice_mut(s![.., 1]).assign(&array![10000., 10000. * line_1[0] + line_1[1]]);
            let x = seg2poly(&s1.view(), &p.view());
            point_ref.slice_mut(s![5, ..]).assign(&x.t());

            println!("point_ref: {point_ref:?}");
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

    let tmp = get_segmentation(data);
}

fn get_segmentation(data: LayoutData) {
    if data.type_ == 11 {
        todo!("Return None");
    }

    let type_data = &ROOM_TYPES[data.type_ as usize];

    let point = data.point;
    let lines = type_data.lines.clone();
    println!("lines: {lines:?}");

    let point = point.mapv(|x| x as i32);
    println!("point: {point:?}");

    let pt1s = point.select(
        Axis(0),
        lines.mapv(|x| x - 1)
            .slice(s![.., 0])
            .to_owned()
            .as_slice()
            .unwrap(),
    );
    println!("lines: {lines:?}");
    println!("pt1s: {pt1s:?}");

    // TODO:
    // pt1s[pt1s <= 0] = 1
    // pt1s[pt1s[:, 0] > data['resolution'][0], 0] = data['resolution'][0]
    // pt1s[pt1s[:, 1] > data['resolution'][1], 1] = data['resolution'][1]
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
        for<'a> &'a T: Deref<Target=T>,
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
    println!("m: {m:?}");

    let b = &s1.slice(s![.., 1]).insert_axis(Axis(1)) - &a;
    let b: Array1<f32> = array![b[(0, 0)], b[(1, 0)]];
    println!("b: {b:?}");

    let x: Array1<f32> = array![b[1], - b[0]].dot(&m);
    println!("x: {x:?}");

    let sx = x.mapv(|x| x.signum());
    println!("sx: {sx:?}");

    let ind: Array1<bool> = (&sx.slice(s![0..-1]) * &sx.slice(s![1..]))
        .mapv(|x| x <= 0.);
    println!("ind: {ind:?}");

    if ind.iter().any(|&x| x) {
        println!("Any!");
        let ind: Vec<usize> = ind.indexed_iter()
            .filter(|(index, &x)| x)
            .map(|(index, _)| index)
            .collect();
        println!("ind: {ind:?}");

        // TODO: this doesn't actually match the original python code as there 2D array is created and selected
        let ind_t: Vec<usize> = ind.iter().map(|x| *x + 1).collect();
        let x1 = x.clone().select(Axis(0), &ind);
        let x2 = x.clone().select(Axis(0), &ind_t);
        println!("x1: {x1:?}");
        println!("x2: {x2:?}");

        let d: Array1<f32> = &b / (b[0].powi(2) + b[1].powi(2));
        println!("d: {d:?}");

        let m_ind = m.select(Axis(1), &ind);
        println!("m_ind: {m_ind:?}");
        let y1 = d.dot(&m.select(Axis(1), &ind));
        let y2 = d.dot(&m.select(Axis(1), &ind_t));
        println!("y1: {y1:?}");
        println!("y2: {y2:?}");

        let dx: Array1<f32> = &x2 - &x1;
        println!("dx: {dx:?}");

        // We won't bother with the degenerate case of dx=0 and x1=0
        let y: Array1<f32> = (&y1 * &x2 - &y2 * &x1) / &dx;
        println!("y: {y:?}");

        // Check if the cross point is inside the segment
        let ind: Array1<bool> = y.mapv(|y| y >= 0. && y < 1.);
        println!("ind: {ind:?}");

        if ind.iter().any(|&x| x) {
            println!("aaa");
            let where_ind: Vec<usize> = ind.indexed_iter()
                .filter(|(index, &x)| x)
                .map(|(index, _)| index)
                .collect();
            // TODO: not sure if this `remove_axis(Axis(1))` acts the same as unsqueeze in all cases
            let x: Array1<f32> = &a.remove_axis(Axis(1)) + &b * &y.select(Axis(0), &where_ind);
            println!("x: {x:?}");
            return x;
        } else {
            println!("bbb");
            let x = Array1::<f32>::zeros((2, ));
            println!("x: {x:?}");
            return x;
        }
    } else {
        println!("ccc");
        let x = Array1::<f32>::zeros((2, ));
        println!("x: {x:?}");
        return x;
    }
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
        let data = vec![vec![1, 2], vec![3, 4], vec![5, 6]];

        let arr = Array2::from_shape_vec((3, 2), data.iter().flatten().collect()).unwrap();
        println!("{arr:?}");
    }

    #[test]
    fn seg2poly_computes_polygon() {
        let t_s1 = array![[157., -100.], [331., 530.8889]];
        let t_p = array![[0., 0., 512.01, 512.01, 0.], [0., 512.01, 512.01, 0., 0.]];

        let res = seg2poly(&t_s1.view(), &t_p.view());
        println!("res: {:?}", res);
    }
}
