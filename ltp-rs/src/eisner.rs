use num_traits::Float;
use std::cmp::{max, min};
use std::fmt::Display;

fn fill<T: Copy>(array: &mut Vec<T>, num: T, size: usize) {
    for i in 0..size {
        array[i] = num;
    }
}

fn backtrack(
    p_i: &Vec<usize>,
    p_c: &Vec<usize>,
    i: usize,
    j: usize,
    complete: bool,
    blk_bias: usize,
    head: &mut Vec<usize>,
    remove_root: usize,
) {
    if i == j {
        return;
    }
    if complete {
        let r = p_c[i * blk_bias + j];
        backtrack(p_i, p_c, i, r, false, blk_bias, head, remove_root);
        backtrack(p_i, p_c, r, j, true, blk_bias, head, remove_root);
    } else {
        let r = p_i[i * blk_bias + j];
        head[j - remove_root] = i;
        backtrack(p_i, p_c, min(i, j), r, true, blk_bias, head, remove_root);
        backtrack(
            p_i,
            p_c,
            max(i, j),
            r + 1,
            true,
            blk_bias,
            head,
            remove_root,
        );
    }
}

pub fn eisner<T>(scores: &[T], stn_length: &[usize], remove_root: bool) -> Vec<Vec<usize>>
where
    T: Float + Display,
{
    // scores [b, w, n]
    let batch = stn_length.len();
    let max_stn_len = *stn_length.iter().max().unwrap();
    let score_block_size = max_stn_len * max_stn_len;

    // [b, n, w]
    let mut bs_i = vec![T::neg_infinity(); score_block_size];
    let mut bs_c = vec![T::neg_infinity(); score_block_size];

    let mut bp_i = vec![0; score_block_size];
    let mut bp_c = vec![0; score_block_size];

    let remove_root = remove_root as usize;

    let mut res = Vec::new();
    for b in 0..batch {
        fill(&mut bs_i, T::neg_infinity(), score_block_size);
        fill(&mut bs_c, T::neg_infinity(), score_block_size);
        fill(&mut bp_i, 0, score_block_size);
        fill(&mut bp_c, 0, score_block_size);

        let max_stn_len_use = stn_length[b];
        let bscore_bias = b * score_block_size;

        for k in 0..max_stn_len_use {
            bs_i[k * max_stn_len_use + k] = T::zero();
            bs_c[k * max_stn_len_use + k] = T::zero();
        }

        for w in 1..max_stn_len_use {
            let n = max_stn_len_use - w;
            // I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
            for i in 0..n {
                let j = i + w;
                let mut max_score = T::neg_infinity();
                let mut max_index = 0;
                for r in i..j {
                    let s = bs_c[i * max_stn_len_use + r]
                        + bs_c[j * max_stn_len_use + r + 1]
                        + scores[bscore_bias + i * max_stn_len + j];
                    if s > max_score {
                        max_score = s;
                        max_index = r;
                    }
                }
                bs_i[j * max_stn_len_use + i] = max_score;
                bp_i[j * max_stn_len_use + i] = max_index;
            }
            // I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
            for i in 0..n {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i..j {
                    let s = bs_c[i * max_stn_len_use + r]
                        + bs_c[j * max_stn_len_use + r + 1]
                        + scores[bscore_bias + j * max_stn_len + i];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_i[i * max_stn_len_use + j] = max_score;
                bp_i[i * max_stn_len_use + j] = max_index;
            }
            // C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
            for i in 0..n {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i..j {
                    let s = bs_c[r * max_stn_len_use + i] + bs_i[j * max_stn_len_use + r];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_c[j * max_stn_len_use + i] = max_score;
                bp_c[j * max_stn_len_use + i] = max_index;
            }
            // C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
            for i in 0..n {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i + 1..j + 1 {
                    let s = bs_i[i * max_stn_len_use + r] + bs_c[r * max_stn_len_use + j];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_c[i * max_stn_len_use + j] = max_score;
                bp_c[i * max_stn_len_use + j] = max_index;
            }
            if stn_length[b] != w {
                bs_c[0 * max_stn_len_use + w] = T::neg_infinity();
            }
        }
        let mut b_head = vec![1usize; max_stn_len_use - remove_root];
        backtrack(
            &bp_i,
            &bp_c,
            0,
            max_stn_len_use - 1,
            true,
            max_stn_len_use,
            &mut b_head,
            remove_root,
        );
        res.push(b_head);
    }
    res
}

#[cfg(test)]
mod tests {
    use ndarray_npy::{NpzReader, ReadNpzError};
    use std::fs::File;

    use crate::eisner::eisner;
    use ndarray::{Array1, Array3};

    #[test]
    fn test_eisner() -> Result<(), ReadNpzError> {
        let mut npz = NpzReader::new(File::open("test/eisner.npz").unwrap())?;
        let scores: Array3<f32> = npz.by_name("scores.npy")?;
        let stn_length: Array1<i64> = npz.by_name("stn_length.npy")?;
        let correct: Array1<i64> = npz.by_name("correct.npy")?;

        let stn_length: Vec<usize> = stn_length.iter().map(|&x| x as usize).collect();
        let output = eisner(scores.as_slice().unwrap(), stn_length.as_slice(), true);

        let correct: Vec<usize> = correct.iter().map(|&x| x as usize).collect();
        let output: Vec<usize> = output.iter().flatten().map(|&x| x).collect();

        assert_eq!(correct, output);

        Ok(())
    }
}
