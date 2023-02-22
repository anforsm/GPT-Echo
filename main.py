from encodec import EncodecModel
from encodec.utils import convert_audio
import numpy as np

import torchaudio
import torch

# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

# Load and pre-process the audio waveform
wav, sr = torchaudio.load("./test_24k.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T][B, n_q, T] -  n_q (int): number of codebooks.]

print(codes)
print(codes.size() )

simpleList = []
for i in codes:
    for ii in i:
        simpleList.append("\n")
        for iii in ii:
            iiiNum= iii.numpy()
            simpleList.append(iiiNum)
            simpleList.append(",")


with open(r'listOfDecodes.txt', 'w') as fp:
    for item in simpleList:
        # write each item on a new line
        fp.write("%s" % item)
    print('Done')

# Decode the codes back to audio
with torch.no_grad():
    decoded = model.decode(codes)
decoded = decoded.squeeze(0)

# Save the decoded audio
torchaudio.save("decoded.wav", decoded, model.sample_rate)

#np.savetxt('myReadableText.txt', torch.Tensor(codes).numpy())