import os
import json

def create_blobs(path):
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
                blob = create_blob(prompt, audio_file)
                blobs.append(blob)
    return blobs



def create_blob(prompt, audio_file):
    return {
        "text": prompt,
        "ljud": audio_file,
    }

if __name__ == "__main__":
    blobs = create_blobs("./data/TRAIN/DR1")
    # Save the blobs as a json file
    with open("blobs.json", "w") as f:
        json.dump(blobs, f, indent=4)