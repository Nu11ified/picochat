use anyhow::{Context, Result};
use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;

/// Synchronous, streaming reader that yields text strings from a parquet file column.
///
/// Reads one record batch at a time and buffers the string values, serving them
/// one at a time via [`next_text`](ParquetTextReader::next_text).
pub struct ParquetTextReader {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    column_name: String,
    buffer: VecDeque<String>,
}

impl ParquetTextReader {
    /// Open a parquet file and prepare to read strings from the given column.
    pub fn open(path: impl AsRef<Path>, column_name: &str) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("failed to open parquet file: {}", path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("failed to read parquet metadata: {}", path.display()))?;
        let reader = builder.build()
            .context("failed to build parquet record batch reader")?;

        Ok(Self {
            reader,
            column_name: column_name.to_string(),
            buffer: VecDeque::new(),
        })
    }

    /// Open a FineWeb parquet file (defaults to the `"text"` column).
    pub fn open_fineweb(path: impl AsRef<Path>) -> Result<Self> {
        Self::open(path, "text")
    }

    /// Returns the next document text from the parquet file, or `None` when exhausted.
    pub fn next_text(&mut self) -> Result<Option<String>> {
        // Serve from buffer first
        if let Some(text) = self.buffer.pop_front() {
            return Ok(Some(text));
        }

        // Try to read next batch
        loop {
            match self.reader.next() {
                Some(batch_result) => {
                    let batch = batch_result.context("failed to read record batch")?;
                    let schema = batch.schema();
                    let col_idx = schema
                        .index_of(&self.column_name)
                        .map_err(|e| anyhow::anyhow!(
                            "column '{}' not found in parquet schema: {}",
                            self.column_name,
                            e
                        ))?;
                    let col = batch.column(col_idx);
                    let string_array = col
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow::anyhow!(
                            "column '{}' is not a string column",
                            self.column_name
                        ))?;

                    for i in 0..string_array.len() {
                        if string_array.is_null(i) {
                            continue;
                        }
                        self.buffer.push_back(string_array.value(i).to_string());
                    }

                    // If this batch had strings, return the first one
                    if let Some(text) = self.buffer.pop_front() {
                        return Ok(Some(text));
                    }
                    // Otherwise, try next batch (this one was empty or all nulls)
                }
                None => return Ok(None),
            }
        }
    }
}

/// Read ALL text from a parquet file, joined by newlines.
pub fn read_all_text(path: impl AsRef<Path>, column_name: &str) -> Result<String> {
    let mut reader = ParquetTextReader::open(path, column_name)?;
    let mut texts = Vec::new();
    while let Some(text) = reader.next_text()? {
        texts.push(text);
    }
    Ok(texts.join("\n"))
}

/// Convert a plain text file to a parquet file with a "text" column.
/// Splits the input on double newlines to create one row per paragraph.
pub fn text_to_parquet(input: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<()> {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let raw = std::fs::read_to_string(input.as_ref())
        .with_context(|| format!("failed to read {}", input.as_ref().display()))?;

    let paragraphs: Vec<&str> = raw
        .split("\n\n")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let array = StringArray::from(paragraphs);
    let num_paragraphs = array.len();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)])?;

    if let Some(parent) = output.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(output.as_ref())
        .with_context(|| format!("failed to create {}", output.as_ref().display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    println!("{} paragraphs written to {}", num_paragraphs, output.as_ref().display());
    Ok(())
}
