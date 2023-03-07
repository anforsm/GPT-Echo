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


def prepare_dataset(batch):
  """Function to preprocess the dataset with the .map method"""
  transcription = batch["sentence"]
  
  if transcription.startswith('"') and transcription.endswith('"'):
    # we can remove trailing quotation marks as they do not affect the transcription
    transcription = transcription[1:-1]
  
  if transcription[-1] not in [".", "?", "!"]:
    # append a full-stop to sentences that do not end in punctuation
    transcription = transcription + "."
  
  batch["sentence"] = transcription
  return batch

index = 0

TOKENIZATION_N_JOBS = 10

datasets.logging.set_verbosity(datasets.logging.ERROR)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens([f"audio_token_{i}" for i in range(1024)])
model.resize_token_embeddings(len(tokenizer))

dataset_train = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train")
dataset_train.map(prepare_dataset, desc="preprocess dataset")
dataset_val = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation")
dataset_val.map(prepare_dataset, desc="preprocess dataset")
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(1.5)

def tokenize_audio(audio_file_og, dataset_type="train"):
    global index
    # Append /en_train_0/ to the last directory of the path
    try:
        audio_file = os.path.join(
            os.path.dirname(audio_file_og), "en_" + dataset_type + "_" + str(index), os.path.basename(audio_file_og)
        )

        wav, sr = torchaudio.load(audio_file)
    except:
        for i in range(1, 24):
            try:
                audio_file = os.path.join(
                    os.path.dirname(audio_file_og), "en_" + dataset_type + "_" + str(i), os.path.basename(audio_file_og)
                )

                wav, sr = torchaudio.load(audio_file)
                index = i
                break
            except:
                continue

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

def filter_function(example):
    return example["up_votes"] > 2 and example["down_votes"] < 2

def main():
    global index
    datasets = DatasetDict({"train": dataset_train, "validation": dataset_val})
    datasets = datasets.filter(filter_function, num_proc=TOKENIZATION_N_JOBS, desc="Filtering dataset")
    # Print number of samples in train dataset
    print(len(datasets["train"]))
    datasets["train"] = datasets["train"].map(lambda example: pad_function(tokenize_function(example)), batch_size=10, num_proc=TOKENIZATION_N_JOBS, remove_columns=datasets["train"].column_names, desc="Padding and Tokenizing training dataset")
    index = 0
    datasets["validation"] = datasets["validation"].map(lambda example: pad_function(tokenize_function(example, "dev")), batch_size=10, num_proc=TOKENIZATION_N_JOBS, remove_columns=datasets["validation"].column_names, desc="Padding and Tokenizing validation dataset")
    datasets.save_to_disk("CV11_distilgpt2_3U_0D_100%.hf")

    trainer_args = TrainingArguments(
        f"{model_name}-finetuned-common-voice",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        push_to_hub=True,
        save_steps=5000
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )

    trainer.train()

    trainer.push_to_hub("trained on samples with more than 2 upvotes and no downvotes")

    #tokenizer.push_to_hub(f"{model_name}-finetuned-common-voice", commit_message="tokenizer with audio tokens")

if __name__ == "__main__":
    main()