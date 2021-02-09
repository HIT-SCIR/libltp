use std::cmp::{min, max};
use num_traits::Float;


fn fill<T: Copy>(array: &mut Vec<T>, num: T, size: usize) {
    for i in 0..size {
        array[i] = num;
    }
}


fn backtrack(p_i: &Vec<usize>, p_c: &Vec<usize>, head: &mut Vec<usize>, bias: usize, i: usize, j: usize, complete: bool, length: usize) {
    if i == j { return; }
    if complete {
        let r = p_c[i * length + j];
        backtrack(p_i, p_c, head, bias, i, r, false, length);
        backtrack(p_i, p_c, head, bias, r, j, true, length);
    } else {
        let r = p_i[i * length + j];
        head[bias + j] = i;
        backtrack(p_i, p_c, head, bias, min(i, j), r, true, length);
        backtrack(p_i, p_c, head, bias, max(i, j), r + 1, true, length);
    }
}

pub fn eisner<T>(scores: &[T], stn_length: &[usize], result: &mut Vec<usize>)
    where T: Float
{
    let batch = stn_length.len();
    let length = *stn_length.iter().max().unwrap();
    let score_size = length * length;

    let mut bs_i = vec![T::min_value(); score_size];
    let mut bs_c = vec![T::min_value(); score_size];

    let mut bp_i = vec![0; score_size];
    let mut bp_c = vec![0; score_size];

    for b in 0..batch {
        fill(&mut bs_i, T::neg_infinity(), score_size);
        fill(&mut bs_c, T::neg_infinity(), score_size);
        fill(&mut bp_c, 0, score_size);
        fill(&mut bp_i, 0, score_size);

        for k in 0..length {
            bs_c[k * length + k] = T::zero();
        }

        let bscore_bias = b * score_size;
        for w in 1..length {
            for i in 0..(length - w) {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i..j {
                    let s = bs_c[i * length + r] + bs_c[j * length + r + 1] + scores[bscore_bias + i * length + j];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_i[j * length + i] = max_score;
                bp_i[j * length + i] = max_index;
            }
            for i in 0..(length - w) {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i..j {
                    let s = bs_c[i * length + r] + bs_c[j * length + r + 1] + scores[bscore_bias + j * length + i];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_i[i * length + j] = max_score;
                bp_i[i * length + j] = max_index;
            }
            for i in 0..(length - w) {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i..j {
                    let s = bs_c[r * length + i] + bs_i[j * length + r];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_c[j * length + i] = max_score;
                bp_c[j * length + i] = max_index;
            }
            for i in 0..(length - w) {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i + 1..j + 1 {
                    let s = bs_i[i * length + r] + bs_c[r * length + j];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_c[i * length + j] = max_score;
                bp_c[i * length + j] = max_index;
            }
            if stn_length[b] != w {
                bs_c[0 * length + w] = T::min_value();
            }
        }
        backtrack(&bp_i, &bp_c, result, b * length, 0, stn_length[b] - 1, true, length);
    }
}

#[cfg(test)]
mod tests {
    use crate::eisner::eisner;


    #[test]
    fn test_eisner() {
        let score = vec![
            0.2166, 0.1328, 0.1851, 0.1822, 0.2832,
            0.2654, 0.1717, 0.1959, 0.1336, 0.2333,
            0.2110, 0.1308, 0.3034, 0.2314, 0.1234,
            0.1694, 0.1963, 0.1647, 0.2776, 0.1920,
            0.2531, 0.1442, 0.1357, 0.2117, 0.2553
        ];
        let mut output = vec![1, 1, 1, 1, 1];
        eisner(&score, &[5], &mut output);
    }
}
