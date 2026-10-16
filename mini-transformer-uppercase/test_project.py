import tempfile
import math
from pathlib import Path
import torch
from dataset import generate
from tokenizer import CharTokenizer
from model import UppercaseTransformer
from train import TextPairs, collate, run_epoch
from inference import predict

def main():
    torch.manual_seed(0)
    tok = CharTokenizer()
    assert tok.decode(tok.encode('hello WORLD')) == 'hello WORLD'
    with tempfile.TemporaryDirectory() as tmp:
        generate(tmp, 12, 4, 4)
        dataset = TextPairs(Path(tmp)/'train.jsonl', tok)
        src, tgt = collate([dataset[0], dataset[1]])
        model = UppercaseTransformer(tok.vocab_size, d_model=32, nhead=4, layers=1, ff_dim=64)
        logits = model(src, tgt[:, :-1])
        assert logits.shape == (2, tgt.size(1)-1, tok.vocab_size)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, tok.vocab_size), tgt[:, 1:].reshape(-1), ignore_index=0)
        loss.backward()
        # Validation runs in eval mode without gradients, which previously
        # activated unsupported nested-tensor operations on MPS.
        devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
        for device in devices:
            model.to(device)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=tok.pad_id)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
            train_loss, _ = run_epoch(model, [(src, tgt)], optimizer, criterion, device)
            val_loss, val_acc = run_epoch(model, [(src, tgt)], None, criterion, device)
            assert math.isfinite(train_loss) and math.isfinite(val_loss)
            assert 0.0 <= val_acc <= 1.0
            print(f'PASS: training and validation on {device}')
        model.cpu()
        result = predict(model, tok, 'hello', 'cpu', max_new_tokens=5)
        assert isinstance(result, str)
        print('PASS: tokenizer, dataset, model, backward pass, inference (untrained)')

if __name__ == '__main__':
    main()
