import argparse
import json
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizer import CharTokenizer
from model import UppercaseTransformer

class TextPairs(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(line) for line in open(path, encoding='utf-8')]
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return (torch.tensor(self.tok.encode(row['input']), dtype=torch.long),
                torch.tensor(self.tok.encode(row['target'], add_bos=True), dtype=torch.long))

def collate(batch):
    src, tgt = zip(*batch)
    return pad_sequence(src, batch_first=True, padding_value=0), pad_sequence(tgt, batch_first=True, padding_value=0)

def run_epoch(model, loader, optimizer, criterion, device):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens, correct = 0., 0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.set_grad_enabled(training):
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        mask = tgt_out.ne(0)
        tokens = int(mask.sum())
        total_loss += loss.item() * tokens
        total_tokens += tokens
        correct += int(((logits.argmax(-1) == tgt_out) & mask).sum())
    return total_loss / max(total_tokens, 1), correct / max(total_tokens, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    p.add_argument('--checkpoint', default='checkpoints/best.pt')
    args = p.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    tok = CharTokenizer()
    train_loader = DataLoader(TextPairs(Path(args.data)/'train.jsonl', tok), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(TextPairs(Path(args.data)/'val.jsonl', tok), batch_size=args.batch_size, collate_fn=collate)
    config = dict(vocab_size=tok.vocab_size, pad_id=tok.pad_id, d_model=args.d_model, layers=args.layers)
    model = UppercaseTransformer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    best = float('inf')
    print(f'Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}')
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, None, criterion, device)
        print(f'Epoch {epoch:02d} train loss={train_loss:.4f} acc={train_acc:.2%} | val loss={val_loss:.4f} acc={val_acc:.2%}', flush=True)
        if val_loss < best:
            best = val_loss
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch}, args.checkpoint)
    print(f'Best checkpoint: {args.checkpoint}')

if __name__ == '__main__':
    main()
