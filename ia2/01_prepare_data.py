from common import *
splits = build_data()
print({k:len(v) for k,v in splits.items()})
print("data + tokenizer written to data/")
