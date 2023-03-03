from tokenizer import AudioTokenizer, Translator
from encodec import EncodecModel


translator = Translator()
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(3)
experimental_audio_tokenizer = AudioTokenizer(model)

experimental_audio_tokenizer.detokenize(translator.emojiTs, 'testemoji_sound.wav')
#print(translator.to_audiotokens(translator.emojiTs))

