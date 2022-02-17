from tokenizers import ByteLevelBPETokenizer
import os

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["data/train.csv", "data/valid.csv", "data/test.csv"],
                vocab_size=52000, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<code>"])
if(os.path.exists("./tokenize") == False):
    os.makedirs("./tokenize")
tokenizer.save_model("./tokenize")