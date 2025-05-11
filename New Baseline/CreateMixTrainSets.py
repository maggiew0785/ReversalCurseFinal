import json
import random

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def make_split(p2d, d2p, p2d_n, d2p_n, seed=42):
    random.seed(seed)
    p2d_sample = random.sample(p2d, p2d_n)
    d2p_sample = random.sample(d2p, d2p_n)
    mixed = p2d_sample + d2p_sample
    random.shuffle(mixed)
    return mixed

def main():
    # load your full train pools
    p2d = load_jsonl('chat_p2d_train.jsonl')
    d2p = load_jsonl('chat_d2p_train.jsonl')

    # define each split: (p2d_count, d2p_count)
    splits = {
        'p2d50': (500, 500),
        'p2d75': (750, 250),
        'p2d25': (250, 750),
        'p2d80': (800, 200),
        'p2d20': (200, 800),
        'p2d90': (900, 100),
        'p2d10': (100, 900),
        'p2d60': (600, 400)
    }

    for name, (p2d_n, d2p_n) in splits.items():
        subset = make_split(p2d, d2p, p2d_n, d2p_n)
        write_jsonl(subset, f'{name}_train.jsonl')

if __name__ == '__main__':
    main()
