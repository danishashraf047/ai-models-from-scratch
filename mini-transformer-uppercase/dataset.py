import argparse
import json
import random
from pathlib import Path

WORDS = '''hello world danish how are you today this is my first neural network model learning machine text transformer python pytorch train data computer science good morning evening friend project small large new example sentence artificial intelligence build code software engineer test model can learn from examples please write simple words we love programming cats dogs birds trees cars roads schools books people work together and create useful things with technology welcome home thank very much what where when why who your our their green blue red quick brown fox jumps over lazy dog beautiful sunny weather coffee tea water food city country language system network deep future smart happy open source research experiment generation token attention decoder encoder training validation accuracy loss input output sequence'''.split()

def make_split(count, seed, used):
    rng = random.Random(seed)
    result = []
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError('Could not create enough unique examples')
        length = rng.randint(2, 7)
        text = ' '.join(rng.choices(WORDS, k=length))
        if len(text) > 55 or text in used:
            continue
        used.add(text)
        result.append({'input': text, 'target': text.upper()})
    return result

def generate(output='data', train_size=12000, val_size=1200, test_size=1200, seed=42):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    used = set()
    for name, count, offset in [('train', train_size, 0), ('val', val_size, 1), ('test', test_size, 2)]:
        rows = make_split(count, seed + offset, used)
        with (path / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
        print(f'Wrote {len(rows)} examples to {path / (name + ".jsonl")}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data')
    p.add_argument('--train-size', type=int, default=12000)
    p.add_argument('--val-size', type=int, default=1200)
    p.add_argument('--test-size', type=int, default=1200)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    generate(args.output, args.train_size, args.val_size, args.test_size, args.seed)
