from datasets import load_dataset, Dataset
import datasets
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from encodec import EncodecModel
from encodec.utils import convert_audio
import torch
import torchaudio
import os
from tqdm import tqdm

TOKENIZATION_N_JOBS = 4

datasets.logging.set_verbosity(datasets.logging.ERROR)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens([f"audio_token_{i}" for i in range(1024)])
model.resize_token_embeddings(len(tokenizer))

dataset_train = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train[:100]")
dataset_val = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:10]")

encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(1.5)

def tokenize_audio(audio_file, dataset_type="train"):
    # Append /en_train_0/ to the last directory of the path
    audio_file = os.path.join(
        os.path.dirname(audio_file), "en_" + dataset_type + "_0", os.path.basename(audio_file)
    )

    wav, sr = torchaudio.load(audio_file)
    wav = torchaudio.functional.vad(wav, sr)
    wav = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
    wav = wav.unsqueeze(0)

    with torch.no_grad():
        frames = encodec_model.encode(wav)
    frame = frames[0][0][0]

    number_of_codebooks, number_of_samples = frame.shape

    tokens = []
    for sample in range(number_of_samples):
        for codebook in range(number_of_codebooks):
            token = frame[codebook, sample].tolist()
            tokens.append(token)

    return tokens

def tokenize_function(example, dataset_type="train"):
    tokenized_audio = tokenize_audio(example["path"], dataset_type)
    if len(tokenized_audio) > 950:
        tokenized_audio = tokenized_audio[:950]
    # Example has client_id, path, sentence etc.
    return tokenizer(
        f"text: {example['sentence']}\nsound: " + 
        "".join(
            [f"audio_token_{token}" for token in tokenized_audio]
        )
    )

def pad_function(example):
    padded_inputs = example["input_ids"] + [tokenizer.pad_token_id] * (1024 - len(example["input_ids"]))
    padded = {}
    padded["input_ids"] = padded_inputs
    padded["attention_mask"] = example["attention_mask"] + [0] * (1024 - len(example["attention_mask"]))
    padded["labels"] = padded_inputs 
    return padded 

def main():
    datasets = DatasetDict({"train": dataset_train, "validation": dataset_val})
    datasets["train"] = datasets["train"].map(lambda example: pad_function(tokenize_function(example)), batch_size=10, num_proc=TOKENIZATION_N_JOBS, remove_columns=datasets["train"].column_names, desc="Padding and Tokenizing training dataset")
    datasets["validation"] = datasets["validation"].map(lambda example: pad_function(tokenize_function(example, "dev")), batch_size=10, num_proc=TOKENIZATION_N_JOBS, remove_columns=datasets["validation"].column_names, desc="Padding and Tokenizing validation dataset")

    trainer_args = TrainingArguments(
        f"{model_name}-finetuned-common-voice",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        #push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )

    trainer.train()

    #trainer.push_to_hub("trained on 100 samples")

    #tokenizer.push_to_hub(f"{model_name}-finetuned-common-voice", commit_message="tokenizer with audio tokens")

if __name__ == "__main__":
    main()