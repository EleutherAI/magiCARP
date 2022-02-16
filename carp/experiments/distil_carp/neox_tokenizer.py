import json
tokenizer_data = None
with open("20B_tokenizer.json") as f:
    tokenizer_data = json.load(f)
line_break_token_ids = list()
for idx, k in enumerate(tokenizer_data['model']['vocab'].keys()):
    if u"\u010a" in k:
        line_break_token_ids.append(idx)
