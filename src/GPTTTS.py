from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, AutoConfig
from encodec import EncodecModel
from encodec.utils import convert_audio
import torch
import torchaudio
import re

class GPTTTS(PreTrainedModel):
    def __init__(self, *model_args, **model_kwargs):
        super().__init__(AutoConfig.from_pretrained("Ekgren/distilgpt2-finetuned-common-voice"), *model_args, **model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained("Ekgren/distilgpt2-finetuned-common-voice")
        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.encodec_model.set_target_bandwidth(1.5)
        self.sample_rate = self.encodec_model.sample_rate
    
    def forward(self, input_ids):
        #decoded = tokenizer.decode(tokens[0], skip_special_tokens=True)
        #decoded = input_text
        # Get all audio_token_
        #pattern = r'audio_token_(\d+)'
        #audio_tokens = re.findall(pattern, decoded)
        #audio_tokens = [int(token) for token in audio_tokens]

        tokens = self.model.generate(input_ids, do_sample=True, max_length=1024, temperature=1, top_k=50, top_p=0.95)[0]
        print(tokens)
        # Get all tokens which are larger than 50257, and subtract 50257 from them
        audio_tokens = [token - 50257 for token in tokens if token > 50257]
        print(audio_tokens)

        number_of_codebooks = 2
        number_of_samples = len(audio_tokens) // number_of_codebooks
        frame = torch.zeros(1, number_of_codebooks, number_of_samples, dtype=torch.long)
        for sample in range(number_of_samples):
            for codebook in range(number_of_codebooks):
              frame[0, codebook, sample] = audio_tokens[sample * number_of_codebooks + codebook]
    
        frames = [(frame, None)]

        with torch.no_grad():
            wav = self.encodec_model.decode(frames)
    
        return wav[0, :, :]


class GPTTTSTokenizer(PreTrainedTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("anforsm/distilgpt2-finetuned-common-voice")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def tokenize(self, text, *args, **kwargs):
        prompt = f"text: {text}\nsound:"
        return self.tokenizer(prompt, return_tensors="pt")
    
    def _tokenize(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)
    
    def convert_tokens_to_ids(self, tokens):
        return tokens["input_ids"]
    
    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)
    
    def _batch_encode_plus(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)
    
    def _encode_plus(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)
    
    
    def save_vocabulary(self, *args, **kwargs):
        return self.tokenizer.save_vocabulary(*args, **kwargs)
    

    

