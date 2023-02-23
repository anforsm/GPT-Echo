from encodec import EncodecModel
from audio_tokenizer import AudioTokenizer


model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6)

tokenizer = AudioTokenizer(model, 64000)
#audio_tokens = tokenizer.tokenize("test_24k.wav")
audio_tokens = tokenizer.tokenize("./data/TRAIN/DR1/FCJF0/SA1.WAV")
tokenizer.detokenize(audio_tokens, "decoded.wav")

def tokenize_text(prompt):
    return [1, 2, 3]

def convert_blob_to_tokens(blob):
    prompt = blob["text"]
    audio = blob["ljud"]

    # Convert the audio to tokens
    audio_tokens = tokenizer.tokenize(audio)

    # Convert the prompt to tokens
    text_tokens = tokenize_text("text: " + prompt + "ljud: ")

    return text_tokens + audio_tokens
