import Stemmer
import string

stemmer = Stemmer.Stemmer('english', maxCacheSize=0)

fold_to_ascii = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
punct_trans = str.maketrans({key: ' ' for key in string.punctuation})
all_trans = {**fold_to_ascii, **punct_trans}


def stem_word(word):
    return stemmer.stemWord(word)


def snowball_tokenizer(text):
    text = text.translate(all_trans)
    split = text.lower().split()
    return [stem_word(token)
            for token in split]


def ws_tokenizer(text):
    text = text.translate(all_trans)
    split = text.lower().split()
    return split
