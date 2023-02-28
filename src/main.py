from tokenizer import Tokenizer
from blobmanager import BlobManager


tokenizer = Tokenizer()
blobmanager = BlobManager()

def test_decode():
    tokenizer.audio_tokenizer.encodec_model.set_target_bandwidth(1.5)
    tokens = tokenizer.tokenize("./../data/CV/cv-corpus-12.0-delta-2022-12-07/en/clips/common_voice_en_35095766.mp3", "test")
    print(len(tokens))
    tokenizer.detokenize(tokens, "./outputs/test_decode.wav", "./outputs/test_decode.txt")

def create_blobs():
    blobs = blobmanager.create_blobs()
    blobmanager.save_blobs_to_file(blobs, "src/outputs/blobs.json")
    #blobmanager.save_blobs_to_file(blobmanager.tokenize_blobs(tokenizer, blobs), "./outputs/blobs_tokenized.json")

def test_blobs_cv():
    blobs = blobmanager.create_blobs_CV()
    #blobmanager.save_blobs_to_file(blobs, "./outputs/blobs.json")
    #blobmanager.save_blobs_to_file(blobmanager.tokenize_blobs(tokenizer, blobs), "./outputs/blobs_tokenized.json")
    #print(blobs)

def downloader_CV_data():
    from huggingface_hub import hf_hub_download
    #hf_hub_download(repo_id="mozilla-foundation/common_voice_11_0", filename="fleurs.py", repo_type="dataset")
    #blobmanager.save_blobs_to_file(blobs, "./outputs/blobs.json")
    #blobmanager.save_blobs_to_file(blobmanager.tokenize_blobs(tokenizer, blobs), "./outputs/blobs_tokenized.json")
    #print(blobs)

def test_decode_blob():
    tokens = blobmanager.load_blobs_from_file("./outputs/blobs_tokenized.json")[213]
    tokenizer.detokenize(tokens, "./outputs/test_decode_blob.wav", "./outputs/test_decode_blob.txt")

if __name__ == "__main__":
    create_blobs()
