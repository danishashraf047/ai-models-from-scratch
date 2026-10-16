import argparse
import torch
from tokenizer import CharTokenizer
from model import UppercaseTransformer

@torch.no_grad()
def predict(model, tok, text, device, max_new_tokens=80):
    if not text:
        return ''
    src = torch.tensor([tok.encode(text)], device=device)
    generated = torch.tensor([[tok.bos_id]], device=device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(src, generated)
        next_id = int(logits[0, -1].argmax())
        if next_id == tok.eos_id:
            break
        generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=1)
    return tok.decode(generated[0].tolist())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--text', required=True)
    p.add_argument('--checkpoint', default='checkpoints/best.pt')
    p.add_argument('--device', default='cpu', choices=['cpu','cuda','mps'])
    args = p.parse_args()
    device = torch.device(args.device)
    tok = CharTokenizer()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UppercaseTransformer(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    if len(args.text) + 1 > 128:
        raise ValueError('Input exceeds maximum 127 characters')
    print('Input: ', args.text)
    print('Model: ', predict(model, tok, args.text, device))
    print('Target:', args.text.upper(), '(for comparison only)')

if __name__ == '__main__':
    main()
