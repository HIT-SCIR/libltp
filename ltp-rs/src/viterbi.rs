use num_traits::{PrimInt};


pub fn viterbi_decode_postprocess<T>(history: &[T], last_tags: &[T], stn_lengths: &[usize], labels_num: usize) -> Vec<Vec<T>>
    where T: PrimInt
{
    // history
    // crf start/end = 2
    // have saved end = 1
    // (stn_num - 1) * stn_len * labels_num

    let stn_length_iter = stn_lengths.iter();
    let &max_stn_len = stn_length_iter.max().unwrap();
    let stn_bias = max_stn_len * labels_num;

    let mut result: Vec<Vec<T>> = Vec::new();
    let mut stn_idx = 0;
    for &stn_len in stn_lengths {
        for _search_idx in 0..stn_len {
            let best_last_tag = last_tags[stn_idx];
            let mut best_tags = vec![best_last_tag];

            // history
            // crf start/end = 2
            // have saved end = 1
            // (stn_num - 1) * stn_len * labels_num
            let path_bias = max_stn_len - stn_len;

            for search_end in 1..(stn_len) { // last one has been used
                let search_end = stn_len - 1 - search_end + path_bias; // 2 is start and end
                // println!("{} {}", stn_idx, search_end);
                let forward_best = *best_tags.last().unwrap();
                let index = search_end * stn_bias + stn_idx * labels_num + forward_best.to_usize().unwrap();
                let last_best = history[index];
                best_tags.push(last_best);
            }
            best_tags.reverse();
            result.push(best_tags);
            stn_idx = stn_idx + 1;
        }
    }
    result
}
