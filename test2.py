from encodec import EncodecModel
from audio_tokenizer import AudioTokenizer
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json

TEXT_VOCAB_SIZE = 65000

tokenizer_text = AutoTokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-126m")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6)
tokenizer = AudioTokenizer(model, TEXT_VOCAB_SIZE)

def test_tokenizer():
    #audio_tokens = tokenizer.tokenize("test_24k.wav")
    audio_tokens = tokenizer.tokenize("./data/TRAIN/DR1/FCJF0/SA1.WAV")
    tokenizer.detokenize(audio_tokens, "decoded.wav")



def tokenize_text(prompt):
    input_ids = tokenizer_text(prompt, return_tensors="pt")["input_ids"].to(device)
    tokens = input_ids.tolist()[0]
    return tokens

def convert_blob_to_tokens(blob):
    prompt = blob["text"]
    audio = blob["ljud"]

    # Convert the audio to tokens
    audio_tokens = tokenizer.tokenize(audio)

    # Convert the prompt to tokens
    text_tokens = tokenize_text(prompt)

    return text_tokens + audio_tokens

def decode_tokenized_blob(tokens):
    # Split the tokens into text and audio tokens
    text_tokens = [] 
    audio_tokens = [] 
    for token in tokens:
        if token < TEXT_VOCAB_SIZE:
            text_tokens.append(token)
        else:
            audio_tokens.append(token)

    # Decode the text tokens
    text = tokenizer_text.decode(text_tokens)

    # Decode the audio tokens
    tokenizer.detokenize(audio_tokens, "decoded.wav")

    return text

def tokenize_blobs():
    # Load the blobs
    with open("blobs.json", "r") as f:
        blobs = json.load(f)
    
    # Convert the blobs to tokens
    tokens = [convert_blob_to_tokens(blob) for blob in blobs]

    # Save the tokens as a json file
    with open("tokens.json", "w") as f:
        json.dump(tokens, f, indent=4)
    
    print(f"Saved {len(tokens)} tokenized prompts and audio files to tokens.json")



if __name__ == "__main__":
    #tokenize_blobs()

    with open("tokens.json", "r") as f:
        tokens = json.load(f)

    text = decode_tokenized_blob(tokens[201])
    print(text)