import os
import wave
from sphfile import SPHFile

def get_metadata(audio_file, is_TIMIT_file=False):
  if not audio_file.split(".")[-1].lower() == "wav":
    raise Exception("Only wav files are supported")
  
  if is_TIMIT_file:
    sph = SPHFile(audio_file)
    audio_file = "./test.wav"
    sph.write_wav(audio_file)

  with wave.open(audio_file, "rb") as f:
    nchannels, sampwidth, framerate, nframes, comptype, compname = f.getparams()
    return {
      "nchannels": nchannels,
      "sampwidth": sampwidth,
      "framerate": framerate,
      "nframes": nframes,
      "comptype": comptype,
      "compname": compname
    }

def print_metadata(audio_file, is_TIMIT_file=False):
  metadata = get_metadata(audio_file, is_TIMIT_file)
  print(f"Metadata for {audio_file}:")
  for key, value in metadata.items():
    print(f"  {key}: {value}")

if __name__ == "__main__":
  print_metadata("./data/TRAIN/DR1/FCJF0/SA1.WAV", True)
  print_metadata("./test_24k.wav")
  print_metadata("./decoded.wav")
