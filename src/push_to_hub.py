from huggingface_hub import notebook_login

def push_model_to_hub(model, hf_username, model_name="bge-base-my-qna-model"):
    notebook_login()  # requires HF token
    model_id = f"{hf_username}/{model_name}"
    model.save_to_hub(model_id, commit_message="Fine-tuned BGE-base on Q&A dataset")
    return model_id
