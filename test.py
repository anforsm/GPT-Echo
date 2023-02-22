from encodec import EncodecModel
from encodec.utils import convert_audio
import numpy as np
import math
import random

import torchaudio
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

wav, sr = torchaudio.load("./test_24k.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
# Adds a dimension of size 1 at the beginning of the tensor
wav = wav.unsqueeze(0)
# e.g. tensor([[[ 0.0782,  0.0724,  0.0822,  ..., -0.0171, -0.0211, -0.0611]]])

with torch.no_grad():
    frames = model.encode(wav)

audio_length = wav.shape[-1]
print("Audio length: ", audio_length)

num_codebooks = frames[0][0].shape[1]
print("Number of codebooks: ", num_codebooks)

frame_size = frames[0][0].shape[2]
print("Frame size: ", frame_size)

print("Frames shape: ", frames[0][0].shape)
# 1, 8, 1500
# _ = batch size
#    K = number of codebooks
#       T = number of frames

# print the first frame, including frame, codebook and value, with labels
print("First frame:")
for k, value in enumerate(frames[0][0][0, :, 0].tolist()):
    print(f"Codebook {k}, value {value}")

# Each frame contains 8 codebooks, each with a value of up to 1500(?)
# We want to flatten the frames into a single list of values
# so we can save it to a text file



values  = []

for (frame, scale) in frames:
    _, K, T = frame.shape

    for t in range(T):
        for k, value in enumerate(frame[0, :, t].tolist()):
            values.append(value)
            # print(f"Frame {t}, codebook {k}, value {value}")

# print(len(values))
# 12000 values
# Generate 12000 random values between 0 and 1024
values = [random.randint(0, 1023) for i in range(12000)]


# print(values)
# print(len(values))
# print(K * T)

# The audio is represented by a 3 dimensional tensor
# The first dimension is K, the number of codebooks
# The second dimension is T, the number of frames
# The third dimension is the value of a specific codebook in a specific frame

# Convert the list of values to an audio file
# model segment stride = None
this_segment_length = audio_length
frame_length = int(math.ceil(this_segment_length / model.sample_rate * model.frame_rate))
# frame_length = 1500
scale = None
# model.bits_per_codebook = 10
# 2**10 = 1024

frames = []
frame = torch.zeros(1, num_codebooks, frame_length, dtype=torch.long)

for t in range(frame_length):
  for k in range(num_codebooks):
    frame[0, k, t] = values[t * num_codebooks + k]

frames.append((frame, scale))

with torch.no_grad():
    wav = model.decode(frames)

#print(values)

# Save wav to file
torchaudio.save("decoded.wav", wav[0, :, :audio_length], model.sample_rate)





