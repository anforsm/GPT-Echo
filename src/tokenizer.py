import os
from encodec.utils import convert_audio
from encodec import EncodecModel
from transformers import AutoTokenizer
from sphfile import SPHFile
import torchaudio
import torch
from typing import List
import json

class Translator:
    def __init__(self):
        
        with open("inter_vocab.json", 'r') as fin:
            self.vocab = json.load(fin)
        
        self.tokens = [i for i in range(65000,66024)] #ints over 64000? 1024 st iaf
        self.emojiTs = list(self.vocab.values())


        self.translatorE_T = {}
        self.translatorT_E = {}
        zipped = zip(self.emojiTs, self.tokens)
        zippedlist = list(zipped)
        for e,t in zippedlist:
            self.translatorE_T[e] = t
            self.translatorT_E[t] = e

    def to_audiotokens(self,GPToutput):
        tokens = [self.translatorE_T[emoj] for emoj in GPToutput]
        return tokens

    def from_audiotokens(self,encoded):
        tokens = [self.translatorT_E[token] for token in encoded]
        return tokens

class AudioTokenizer:
    bandwidth_to_codebooks = {
        1.5: 2,
        3.0: 4,
        6.0: 8,
        12.0: 16,
        24.0: 32,
    }
    
    def __init__(self, encodec_model, vocab_start=0):
        self.vocab_start = vocab_start
        self.encodec_model = encodec_model
    
    def translate_to_vocab(tokens,vocabfile):
        pass

    def tokenize(self, audio_file):
        if (os.path.basename(audio_file).endswith(".sph") or os.path.basename(audio_file).endswith(".WAV")):
            audio_file = self.convert_SPH_to_wav(audio_file)
        elif (os.path.basename(audio_file).endswith(".wav") or os.path.basename(audio_file).endswith(".mp3")):
            pass
        else:
            raise Exception("File type not supported")

        return self.tokenize_wav(audio_file)
    
    def convert_SPH_to_wav(self, sph_file):
        sph = SPHFile(sph_file)
        new_filename = sph_file + "converted.wav"
        sph.write_wav(new_filename)
        return new_filename

    
    def tokenize_wav(self, audio_file):
        wav, sr = torchaudio.load(audio_file)
        wav = convert_audio(wav, sr, self.encodec_model.sample_rate, self.encodec_model.channels)
        wav = wav.unsqueeze(0)

        # Encode and compress the audio 
        with torch.no_grad():
            frames = self.encodec_model.encode(wav)
        frame = frames[0][0][0]

        number_of_codebooks, number_of_samples = frame.shape

        # Convert encoding to a list of tokens
        tokens = []
        translator = Translator()
        for sample in range(number_of_samples):
            for codebook in range(number_of_codebooks):
                token = frame[codebook, sample].tolist() # tal mellan 0-1023
                tokens.append(self.vocab_start + token) #måste ändras
                translated = translator.from_audiotokens(tokens) #ÄNDRAT

        return translated #ÄNDRAT
    
    def detokenize(self, tokens, audio_file):
        translator = Translator()
        translated = translator.to_audiotokens(tokens)
        number_of_codebooks = AudioTokenizer.bandwidth_to_codebooks[self.encodec_model.bandwidth]
        number_of_samples = len(translated) // number_of_codebooks

        frame = torch.zeros(1, number_of_codebooks, number_of_samples, dtype=torch.long)

        for sample in range(number_of_samples):
            for codebook in range(number_of_codebooks):
                frame[0, codebook, sample] = translated[tokens[sample * number_of_codebooks + codebook]]
        
        frames = [(frame, None)]
        
        with torch.no_grad():
            wav = self.encodec_model.decode(frames)
        
        torchaudio.save(audio_file, wav[0, :, :], self.encodec_model.sample_rate)

class TextTokenizer:
    def __init__(self, tokenizer_model="AI-Sweden-Models/gpt-sw3-126m", vocab_start=0):
        self.vocab_start = vocab_start
        self.tokenizer_model = AutoTokenizer.from_pretrained(tokenizer_model)
    
    def tokenize(self, prompt):
        input_ids = self.tokenizer_model(prompt, return_tensors="pt")["input_ids"]
        tokens = input_ids.tolist()[0]
        return tokens

    def detokenize(self, tokens):
        return self.tokenizer_model.decode(tokens)
    

class Tokenizer:
    def __init__(self):
        self.target_audio_bandwidth = 6
        self.audio_vocab_start = 65000
        self.gpt_model = "AI-Sweden-Models/gpt-sw3-126m"
        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.encodec_model.set_target_bandwidth(self.target_audio_bandwidth)

        self.text_prefix = "text: "
        self.text_suffix = "\n"
        self.audio_prefix = "audio: "


        self.audio_tokenizer = AudioTokenizer(self.encodec_model, self.audio_vocab_start)

        self.text_tokenizer = TextTokenizer(self.gpt_model)
    
    def tokenize(self, audio_file, prompt) -> List[int]:
        audio_tokens = self.audio_tokenizer.tokenize(audio_file)
        text_tokens = self.text_tokenizer.tokenize(
            self.text_prefix + 
            prompt + 
            self.text_suffix +
            self.audio_prefix)

        return text_tokens + audio_tokens
    
    def detokenize(self, tokens, audio_file, prompt_file):
        audio_tokens = []
        text_tokens = []
        for token in tokens:
            if token < self.audio_vocab_start:
                text_tokens.append(token)
            else:
                audio_tokens.append(token)

        self.audio_tokenizer.detokenize(audio_tokens, audio_file)
        text = self.text_tokenizer.detokenize(text_tokens)

        with open(prompt_file, "w") as f:
            f.write(text)

