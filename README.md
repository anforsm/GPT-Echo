GPT-Echo is a TTS-model using the generative GPT-2 model
and using Facebooks Encodec, an audio compression model.

You can test the GPT-Echo model out in a easy-to-use demo:
https://huggingface.co/spaces/anforsm/GPT-TTS-demo

It can also be used directly from the notebook-file main.ipynb.
The model itself is available at:
https://huggingface.co/anforsm/GPT-Echo-82m

The tokenizer can be found at:
https://huggingface.co/anforsm/distilgpt2-finetuned-common-voice/tree/main

The dataset used is a subset of Mozilla Common Voice 11.0
The tokenized dataset can be found at:
https://huggingface.co/datasets/anforsm/common_voice_11_clean_tokenized

CREDIT:
https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
https://huggingface.co/distilgpt2
https://github.com/facebookresearch/encodec
