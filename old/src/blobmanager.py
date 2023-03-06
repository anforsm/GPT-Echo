import os
import json

from datasets import load_dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler

cv_11 = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train")
batch_sampler = BatchSampler(RandomSampler(cv_11), batch_size=32, drop_last=False)
# Save dataloader to later
# dataloader = DataLoader(cv_11, batch_sampler=batch_sampler) 

class BlobManager:
    def __init__(self):
        pass
    
    def create_blobs(self):
        return self.create_blobs_CV()
        #return self.create_blobs_timit()
    
    def create_blob(self, prompt, audio_file):
        return {
            "text": prompt,
            "ljud": audio_file,
        }

    def create_blobs_CV(self):
        blobs = []
        #cv_11.save_to_disk("data/CV")
        # Loop through all data, and extract sentence and audio-path

        
        for i in range(len(cv_11)):
            blob = self.create_blob(cv_11[i]["sentence"], cv_11[i]["audio"]["path"])
            blobs.append(blob)
        '''
        for i in range(3):
            blob = self.create_blob(cv_11[i]["sentence"], cv_11[i]["audio"]["path"])
            blobs.append(blob)
        '''
        #print(blobs)
        return blobs
    
    def create_blobs_timit(self):
        path = "./../data/TRAIN/DR1"
        blobs = []
        # Loop through all folders in the path, each folder is a speaker
        for speaker in os.listdir(path):
            speaker_path = os.path.join(path, speaker)
            # Loop through all files in the speaker folder
            for file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file)
                # If the file is a .wav file, create a blob
                if file_path.endswith(".WAV"):
                    # Create a blob
                    prompt_file = file_path.replace(".WAV", ".TXT")
                    prompt = " ".join(open(prompt_file, "r").read().split(" ")[2:]).replace("\n", "")
                    audio_file = file_path
                    blob = self.create_blob(prompt, audio_file)
                    blobs.append(blob)
        return blobs
    
    def save_blobs_to_file(self, blobs, file):
        with open(file, "w") as f:
            json.dump(blobs, f, indent=4)
    
    def tokenize_blob(self, tokenizer, blob):
        return tokenizer.tokenize(blob["ljud"], blob["text"])
    
    def tokenize_blobs(self, tokenizer, blobs):
        tokens = [self.tokenize_blob(tokenizer, blob) for blob in blobs]
        return tokens
    
    def load_blobs_from_file(self, file):
        with open(file, "r") as f:
            return json.load(f)
