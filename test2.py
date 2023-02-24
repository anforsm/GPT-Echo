from encodec import EncodecModel
from audio_tokenizer import AudioTokenizer
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

tokenizer_text = AutoTokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-126m")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6)

tokenizer = AudioTokenizer(model, 64000)
#audio_tokens = tokenizer.tokenize("test_24k.wav")
audio_tokens = tokenizer.tokenize("./data/TRAIN/DR1/FCJF0/SA1.WAV")
tokenizer.detokenize(audio_tokens, "decoded.wav")

def tokenize_text(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
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
