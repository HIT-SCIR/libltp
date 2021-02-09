fn start_of_chunk(prev_tag: &str, tag: &str, prev_type: &str, type_: &str) -> bool {
    if tag == "B" {
        return true;
    }
    if tag == "S" {
        return true;
    }

    if prev_tag == "E" && tag == "E" {
        return true;
    }
    if prev_tag == "E" && tag == "I" {
        return true;
    }
    if prev_tag == "S" && tag == "E" {
        return true;
    }
    if prev_tag == "S" && tag == "I" {
        return true;
    }
    if prev_tag == "O" && tag == "E" {
        return true;
    }
    if prev_tag == "O" && tag == "I" {
        return true;
    }

    if tag != "O" && tag != "." && prev_type != type_ {
        return true;
    }
    false
}

fn end_of_chunk(prev_tag: &str, tag: &str, prev_type: &str, type_: &str) -> bool {
    if prev_tag == "E" {
        return true;
    }
    if prev_tag == "S" {
        return true;
    }

    if prev_tag == "B" && tag == "B" {
        return true;
    }
    if prev_tag == "B" && tag == "S" {
        return true;
    }
    if prev_tag == "B" && tag == "O" {
        return true;
    }
    if prev_tag == "I" && tag == "B" {
        return true;
    }
    if prev_tag == "I" && tag == "S" {
        return true;
    }
    if prev_tag == "I" && tag == "O" {
        return true;
    }

    if prev_tag != "O" && prev_tag != "." && prev_type != type_ {
        return true;
    }
    false
}

pub fn get_entities(mut seq: Vec<&str>) -> Vec<(&str, usize, usize)> {
    seq.push("O");

    let mut prev_tag = "O";
    let mut prev_type = "_";
    let mut begin_offset: usize = 0;
    let mut chunks: Vec<(&str, usize, usize)> = Vec::new();

    for (i, chunk) in seq.iter().enumerate() {
        let cut = chunk.find('-');
        let (tag, type_) = match cut {
            None => (&chunk[0..], "_"),
            Some(cut) => (&chunk[..cut], &chunk[cut + 1..]),
        };
        if end_of_chunk(prev_tag, tag, prev_type, type_) {
            chunks.push((prev_type.clone(), begin_offset, i - 1));
        }
        if start_of_chunk(&prev_tag, tag, prev_type, type_) {
            begin_offset = i;
        }
        prev_tag = tag;
        prev_type = type_;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use crate::entities::get_entities;

    #[test]
    fn test_get_entities() {
        let example = vec!["B-PER", "I-PER", "O", "B-LOC"];
        let result = get_entities(example);
        assert_eq!(result, vec![("PER", 0, 1), ("LOC", 3, 3)]);
    }
}