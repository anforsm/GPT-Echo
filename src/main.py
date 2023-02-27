from tokenizer import Tokenizer
from blobmanager import BlobManager


tokenizer = Tokenizer()
blobmanager = BlobManager()

def test_decode():
    tokens = tokenizer.tokenize("./../data/TRAIN/DR1/FCJF0/SA1.WAV", "She had your dark suit in greasy wash water all year.")
    print(tokens)
    tokenizer.detokenize(tokens, "./outputs/test_decode.wav", "./outputs/test_decode.txt")

def create_blobs():
    blobs = blobmanager.create_blobs()
    blobmanager.save_blobs_to_file(blobs, "./outputs/blobs.json")
    blobmanager.save_blobs_to_file(blobmanager.tokenize_blobs(tokenizer, blobs), "./outputs/blobs_tokenized.json")

def test_decode_blob():
    tokens = blobmanager.load_blobs_from_file("./outputs/blobs_tokenized.json")[213]
    tokenizer.detokenize(tokens, "./outputs/test_decode_blob.wav", "./outputs/test_decode_blob.txt")

if __name__ == "__main__":
    test_decode_blob()
