use std::sync::Arc;
use arrow::datatypes::{Schema, Field, DataType};

pub fn ltp_result_arrow_scheme() -> Arc<Schema> {
    let dep_struct = DataType::Struct(vec![
        Field::new("arc", DataType::UInt64, false),
        Field::new("rel", DataType::Utf8, false),
    ]);

    let sdp_struct = DataType::Struct(vec![
        Field::new("src", DataType::UInt64, false),
        Field::new("tgt", DataType::UInt64, false),
        Field::new("rel", DataType::Utf8, false),
    ]);

    let struct_fields = vec![
        // list of str
        Field::new("seg", DataType::List(
            Box::from(Field::new("item", DataType::Utf8, true))
        ), true),
        // list of str
        Field::new("pos", DataType::List(
            Box::from(Field::new("item", DataType::Utf8, true))
        ), true),
        // list of str
        Field::new("ner", DataType::List(
            Box::from(Field::new("item", DataType::Utf8, true))
        ), true),
        // list of (list of str)
        Field::new("srl", DataType::List(Box::from(
            Field::new("item", DataType::List(
                Box::from(Field::new("item", DataType::Utf8, true))
            ), true)
        )), true),
        // list of dep
        Field::new("dep", DataType::List(
            Box::from(Field::new("item", dep_struct, true))
        ), true),
        // list of sdp
        Field::new("sdp", DataType::List(
            Box::from(Field::new("item", sdp_struct, true))
        ), true)
    ];
    let schema = Arc::new(Schema::new(struct_fields));
    schema
}

