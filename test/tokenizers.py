import string

fold_to_ascii = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
punct_trans = str.maketrans({key: ' ' for key in string.punctuation})
all_trans = {**fold_to_ascii, **punct_trans}


def ws_tokenizer(text):
    text = text.translate(all_trans)
    split = text.lower().split()
    return split
