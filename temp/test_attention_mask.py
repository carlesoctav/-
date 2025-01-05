from transformers import AutoTokenizer, BertTokenizer


# dummy_tokenizer  = BertTokenizer()
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")


a4, = tokenizer("saya makan nasi")
print(f"DEBUGPRINT[2]: test_attention_mask.py:8: a={a}")
