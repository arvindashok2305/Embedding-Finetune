def fine_tune(model, dataloader, train_loss, output_path="../saved_models/fine-tuned-bge-qna", num_epochs=1):
    warmup_steps = int(len(dataloader) * num_epochs * 0.1)
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True
    )
    return output_path
