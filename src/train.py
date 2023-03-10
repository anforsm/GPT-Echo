from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "distilgpt2"

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens([f"audio_token_{i}" for i in range(1024)])
    model.resize_token_embeddings(len(tokenizer))

    datasets = load_dataset("anforsm/common_voice_11_clean_tokenized")

    trainer_args = TrainingArguments(
        f"anforsm/distilgpt2-finetuned-common-voice",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        push_to_hub=True,
        save_steps=5000,
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )

    trainer.train()
    trainer.push_to_hub("trained on full clean dataset")
