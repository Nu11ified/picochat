use picochat_data::parquet::{ParquetTextReader, read_all_text};

/// Helper to create a test parquet file with a "text" column.
fn create_test_parquet(path: &str, texts: &[&str]) {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::fs::File;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let array = StringArray::from(texts.to_vec());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn test_read_parquet_texts() {
    let path = "/tmp/picochat_test_read_texts.parquet";
    let texts = &["Hello world", "Rust is great", "Parquet works"];
    create_test_parquet(path, texts);

    let mut reader = ParquetTextReader::open(path, "text").unwrap();
    let mut results = Vec::new();
    while let Some(text) = reader.next_text().unwrap() {
        results.push(text);
    }

    assert_eq!(results.len(), 3);
    assert_eq!(results[0], "Hello world");
    assert_eq!(results[1], "Rust is great");
    assert_eq!(results[2], "Parquet works");
}

#[test]
fn test_read_all_text() {
    let path = "/tmp/picochat_test_read_all.parquet";
    let texts = &["First document", "Second document"];
    create_test_parquet(path, texts);

    let combined = read_all_text(path, "text").unwrap();
    assert_eq!(combined, "First document\nSecond document");
}

#[test]
fn test_missing_column_error() {
    let path = "/tmp/picochat_test_missing_col.parquet";
    let texts = &["some text"];
    create_test_parquet(path, texts);

    let mut reader = ParquetTextReader::open(path, "nonexistent").unwrap();
    let result = reader.next_text();
    assert!(result.is_err(), "expected error for missing column");
}
