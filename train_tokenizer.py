from tokenizers import SentencePieceBPETokenizer
tokenizer = SentencePieceBPETokenizer()

tokenizer.train(files=["data/train.csv", "data/valid.csv", "data/test.csv"],
                vocab_size=52000, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<code>"])
tokenizer.save_model("./tokenize")