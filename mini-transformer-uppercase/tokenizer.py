import string

class CharTokenizer:
    def __init__(self):
        chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + ' .,!?-\''
        self.tokens = ['<pad>', '<bos>', '<eos>', '<unk>'] + list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, text, add_bos=False, add_eos=True):
        ids = ([self.bos_id] if add_bos else [])
        ids += [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        return ''.join(self.itos.get(int(i), '') for i in ids if int(i) >= 4)
