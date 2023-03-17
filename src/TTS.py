from transformers import AutoTokenizer, AutoModelForCausalLM
from encodec import EncodecModel
from encodec.utils import convert_audio
import torch
import torchaudio
import re

encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(1.5)
tokenizer = AutoTokenizer.from_pretrained("anforsm/distilgpt2-finetuned-common-voice")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained("Ekgren/distilgpt2-finetuned-common-voice")

def decode(tokens):
    decoded = tokenizer.decode(tokens[0], skip_special_tokens=True)
    # Get all audio_token_
    pattern = r'audio_token_(\d+)'
    audio_tokens = re.findall(pattern, decoded)
    audio_tokens = [int(token) for token in audio_tokens]

    number_of_codebooks = 2
    number_of_samples = len(audio_tokens) // number_of_codebooks
    frame = torch.zeros(1, number_of_codebooks, number_of_samples, dtype=torch.long)
    for sample in range(number_of_samples):
        for codebook in range(number_of_codebooks):
            frame[0, codebook, sample] = audio_tokens[sample * number_of_codebooks + codebook]
    
    frames = [(frame, None)]

    with torch.no_grad():
        wav = encodec_model.decode(frames)
    
    return (wav[0, :, :], encodec_model.sample_rate)


def TTS(text):
    prompt = f"text: {text}\nsound:"
    tokenized = tokenizer(prompt, return_tensors="pt")
    tokens = model.generate(tokenized["input_ids"], do_sample=True, max_length=1024, temperature=1, top_k=50, top_p=0.95)
    return decode(tokens)
