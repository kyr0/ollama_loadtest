#!/usr/bin/env python3
# load_test_ollama_verbose.py
#
# Load test stressing Ollama's KV-cache with logarithmic scaling.
# Prints one concise, information-rich line per request.

import json, time, random, math, argparse
from concurrent.futures import ThreadPoolExecutor
try:
    import urllib.request as ureq
    import urllib.error   as uerr
except ImportError:      # very old 3.x build
    import urllib2 as ureq
    import urllib2 as uerr

URL         = 'http://localhost:11434/api/generate'
HEADERS     = {'Content-Type': 'application/json'}
MAX_TOKENS  = 131072      # 128 k context
BASE_Q      = "What is the capital of France? "
FILLER      = "lorem ipsum "

def build_prompt(n, max_tokens):
    """Build prompt with logarithmic scaling based on request sequence number."""
    # Use logarithmic scaling: log base 2 to grow exponentially but controlled
    # Scale from 1 to max reasonable prompt size (3/4 of max_tokens)
    max_filler_tokens = (max_tokens * 3) // 4
    log_scale = math.log2(n + 2)  # +2 to avoid log(0) and start from a reasonable size
    filler_tokens = min(int(log_scale * 512), max_filler_tokens)  # 512 tokens per log unit
    
    # Generate random noise for cache-busting
    noise = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(16))
    
    # Build the prompt with logarithmically scaled filler
    filler_text = FILLER * (filler_tokens // len(FILLER.split()))
    return BASE_Q + filler_text + noise

def send(seq, model_name, max_tokens):
    print("sending request %05d" % seq)
    prompt = build_prompt(seq, max_tokens)
    payload = {
        "model":  model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": max_tokens}
    }
    data = json.dumps(payload).encode('utf-8')
    req  = ureq.Request(URL, data=data, headers=HEADERS)

    t0 = time.time()
    try:
        with ureq.urlopen(req, timeout=600) as r:
            body = r.read()
            status = r.getcode()
        dt = time.time() - t0
        prompt_tokens = len(prompt.split())
        log_level = math.log2(seq + 2)
        print("Req %05d | %3d | prompt=%6d tok (log2=%.1f) | resp=%5d B | %.2f s"
              % (seq, status, prompt_tokens, log_level, len(body), dt))
    except uerr.URLError as e:
        print("Req %05d | ERR | %s" % (seq, e.reason))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Load test Ollama with logarithmic prompt scaling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-b', '--batch-size', type=int, default=10,
                       help='Number of concurrent requests (batch size)')
    parser.add_argument('-i', '--iterations', type=int, default=100,
                       help='Number of iterations per thread')
    parser.add_argument('-m', '--model', type=str, default='qwen3:0.6b',
                       help='Model name to use for requests')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS,
                       help='Maximum context size in tokens')
    parser.add_argument('--url', type=str, default=URL,
                       help='Ollama API endpoint URL')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting load test with:")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  URL: {args.url}")
    print(f"  Total requests: {args.batch_size * args.iterations}")
    print("  Prompt scaling: Logarithmic (log2-based)")
    print()
    
    global URL
    URL = args.url
    
    total = args.batch_size * args.iterations
    with ThreadPoolExecutor(max_workers=args.batch_size) as ex:
        for i in range(total):
            ex.submit(send, i, args.model, args.max_tokens)

if __name__ == '__main__':
    main()
