import os
from encodec.utils import convert_audio
from sphfile import SPHFile
import torchaudio
import torch

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
    
    def tokenize(self, audio_file):
        if (os.path.basename(audio_file).endswith(".sph") or os.path.basename(audio_file).endswith(".WAV")):
            audio_file = self.convert_SPH_to_wav(audio_file)
        elif (os.path.basename(audio_file).endswith(".wav")):
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
        for sample in range(number_of_samples):
            for codebook in range(number_of_codebooks):
                token = frame[codebook, sample].tolist()
                tokens.append(self.vocab_start + token)

        return tokens
    
    def detokenize(self, tokens, audio_file):
        number_of_codebooks = AudioTokenizer.bandwidth_to_codebooks[self.encodec_model.bandwidth]
        number_of_samples = len(tokens) // number_of_codebooks

        frame = torch.zeros(1, number_of_codebooks, number_of_samples, dtype=torch.long)

        for sample in range(number_of_samples):
            for codebook in range(number_of_codebooks):
                frame[0, codebook, sample] = tokens[sample * number_of_codebooks + codebook] - self.vocab_start
        
        frames = [(frame, None)]
        
        with torch.no_grad():
            wav = self.encodec_model.decode(frames)
        
        torchaudio.save(audio_file, wav[0, :, :], self.encodec_model.sample_rate)




