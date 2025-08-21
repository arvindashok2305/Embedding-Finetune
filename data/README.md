# 📊 Dataset Instructions

This project uses a **Question-Answer dataset** stored in Parquet format for fine-tuning embeddings.

## Download

1. Place your dataset file inside `data/raw/`.
   - Expected file: `train-00000-of-00001.parquet`

2. If you are using a custom dataset, make sure it contains the following columns:
   - `query` (or `question`)
   - `answer`

## Example

