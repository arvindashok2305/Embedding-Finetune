from datasets import load_dataset
from sentence_transformers import InputExample

def load_custom_dataset(file_path="train-00000-of-00001.parquet"):
    dataset = load_dataset("parquet", data_files={"train": file_path}, split="train")

    # Rename column if needed
    if "query" in dataset.column_names and "question" not in dataset.column_names:
        dataset = dataset.rename_column("query", "question")

    return dataset

def prepare_training_examples(dataset, instruction):
    return [
        InputExample(texts=[instruction + record["question"], record["answer"]])
        for record in dataset
    ]
