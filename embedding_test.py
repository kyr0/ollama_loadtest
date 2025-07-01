#!/usr/bin/env python3
# embedding_test_ollama.py
#
# Load test for Ollama's embedding endpoint with logarithmic input scaling.
# Tests embedding generation performance with varying input sizes.

import json, time, random, math, argparse
from concurrent.futures import ThreadPoolExecutor
try:
    import urllib.request as ureq
    import urllib.error   as uerr
except ImportError:      # very old 3.x build
    import urllib2 as ureq
    import urllib2 as uerr

URL         = 'http://localhost:11434/api/embed'
HEADERS     = {'Content-Type': 'application/json'}
BASE_TEXT   = "Llamas are members of the camelid family"
FILLER_SENTENCES = [
    "They are native to South America and are known for their soft wool.",
    "Llamas are highly social animals that live in herds.",
    "They have been domesticated for thousands of years.",
    "These animals are excellent pack animals in mountainous terrain.",
    "Llamas communicate through various vocalizations and body language.",
    "They are herbivores that primarily graze on grass and other plants.",
    "Adult llamas typically weigh between 280 to 450 pounds.",
    "Their wool is prized for its warmth and softness.",
    "Llamas are known for their calm and gentle temperament.",
    "They can carry up to 25-30% of their body weight when used as pack animals."
]

def build_input_text(n, max_length=8192):
    """Build input text with logarithmic scaling based on request sequence number."""
    # Use logarithmic scaling to gradually increase input size
    log_scale = math.log2(n + 2)  # +2 to avoid log(0) and start from reasonable size
    target_chars = min(int(log_scale * 200), max_length)  # 200 chars per log unit
    
    # Start with base text
    text = BASE_TEXT
    
    # Add filler sentences until we reach target length
    while len(text) < target_chars:
        sentence = random.choice(FILLER_SENTENCES)
        text += " " + sentence
        
        # Add some random variation to prevent identical embeddings
        if random.random() < 0.3:  # 30% chance to add noise
            noise = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(8))
            text += f" ({noise})"
    
    # Truncate to target length if needed
    if len(text) > target_chars:
        text = text[:target_chars]
    
    return text

def send_embed_request(seq, model_name, max_length):
    print("sending embed request %05d" % seq)
    input_text = build_input_text(seq, max_length)
    payload = {
        "model": model_name,
        "input": input_text
    }
    data = json.dumps(payload).encode('utf-8')
    req = ureq.Request(URL, data=data, headers=HEADERS)

    t0 = time.time()
    try:
        with ureq.urlopen(req, timeout=600) as r:
            body = r.read()
            status = r.getcode()
        dt = time.time() - t0
        
        # Parse response to get embedding dimension
        try:
            response_data = json.loads(body.decode('utf-8'))
            if 'embeddings' in response_data and response_data['embeddings']:
                embedding_dim = len(response_data['embeddings'][0])
            else:
                embedding_dim = 0
        except (json.JSONDecodeError, KeyError, IndexError):
            embedding_dim = 0
        
        input_chars = len(input_text)
        log_level = math.log2(seq + 2)
        print("Req %05d | %3d | input=%6d chars (log2=%.1f) | dim=%4d | resp=%5d B | %.2f s"
              % (seq, status, input_chars, log_level, embedding_dim, len(body), dt))
    except uerr.URLError as e:
        print("Req %05d | ERR | %s" % (seq, e.reason))
    except Exception as e:
        print("Req %05d | ERR | Unexpected error: %s" % (seq, str(e)))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Load test Ollama embedding endpoint with logarithmic input scaling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-b', '--batch-size', type=int, default=5,
                       help='Number of concurrent requests (batch size)')
    parser.add_argument('-i', '--iterations', type=int, default=50,
                       help='Number of iterations per thread')
    parser.add_argument('-m', '--model', type=str, default='dengcao/Qwen3-Embedding-8B:Q8_0',
                       help='Embedding model name to use for requests')
    parser.add_argument('--max-length', type=int, default=8192,
                       help='Maximum input text length in characters')
    parser.add_argument('--url', type=str, default=URL,
                       help='Ollama embedding API endpoint URL')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting embedding load test with:")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Max input length: {args.max_length} chars")
    print(f"  URL: {args.url}")
    print(f"  Total requests: {args.batch_size * args.iterations}")
    print("  Input scaling: Logarithmic (log2-based)")
    print()
    
    global URL
    URL = args.url
    
    total = args.batch_size * args.iterations
    with ThreadPoolExecutor(max_workers=args.batch_size) as ex:
        for i in range(total):
            ex.submit(send_embed_request, i, args.model, args.max_length)

if __name__ == '__main__':
    main()
