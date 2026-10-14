"""Lab 35: PagedAttention. A mini block allocator, measured against contiguous pre-allocation."""
import random

BLOCK = 16          # tokens per block (vLLM's default is derived per model; 16 is the classic value)
BYTES_PER_TOKEN = 128 * 1024   # pretend 128 KB/token of KV (Llama-3.1-8B is ~128 KB/token in bf16)


class BlockAllocator:
    def __init__(self, total_blocks):
        self.free = list(range(total_blocks))
        self.total = total_blocks
        self.refcount = {}

    def allocate(self, n):
        if n > len(self.free):
            raise MemoryError("no free blocks: this is when vLLM preempts or queues a request")
        got = [self.free.pop() for _ in range(n)]
        for b in got:
            self.refcount[b] = 1
        return got

    def fork(self, block_table):
        """Share the prompt's blocks between two sequences (copy-on-write), as in beam search / n>1."""
        for b in block_table:
            self.refcount[b] += 1
        return list(block_table)

    def free_blocks(self, block_table):
        for b in block_table:
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                self.free.append(b)

    @property
    def used(self):
        return self.total - len(self.free)


class Sequence:
    def __init__(self, alloc, prompt_len):
        self.alloc = alloc
        self.len = prompt_len
        self.blocks = alloc.allocate((prompt_len + BLOCK - 1) // BLOCK)

    def step(self):                      # one generated token
        if self.len % BLOCK == 0 and self.len // BLOCK == len(self.blocks):
            self.blocks += self.alloc.allocate(1)     # grow one block at a time, on demand
        self.len += 1


random.seed(0)
MAX_LEN = 2048                # what the engine would have to reserve per sequence without paging
requests = [(random.randint(50, 400), random.randint(20, 300)) for _ in range(32)]   # (prompt, output)

contiguous = len(requests) * MAX_LEN
paged_peak = 0
alloc = BlockAllocator(total_blocks=10_000)
seqs = [Sequence(alloc, p) for p, _ in requests]
remaining = [o for _, o in requests]
while any(r > 0 for r in remaining):
    for i, s in enumerate(seqs):
        if remaining[i] > 0:
            s.step()
            remaining[i] -= 1
    paged_peak = max(paged_peak, alloc.used * BLOCK)
actual_tokens = sum(p + o for p, o in requests)

print(f"32 requests, prompt 50-400 tokens, output 20-300 tokens, block = {BLOCK} tokens")
print(f"{'strategy':<34}{'KV slots held':>15}{'GB @128KB/token':>18}{'waste':>8}")
for name, slots in [(f"contiguous, reserve max_len={MAX_LEN}", contiguous),
                    ("paged, blocks on demand", paged_peak),
                    ("perfect (tokens actually used)", actual_tokens)]:
    waste = 100 * (slots - actual_tokens) / slots
    print(f"{name:<34}{slots:>15,}{slots * BYTES_PER_TOKEN / 1024 ** 3:>16.1f} GB{waste:>7.1f}%")

print("\nInternal fragmentation with paging is bounded: at most BLOCK-1 wasted slots per sequence")
print(f"  worst case here: 32 x {BLOCK - 1} = {32 * (BLOCK - 1)} slots = "
      f"{32 * (BLOCK - 1) * BYTES_PER_TOKEN / 1024 ** 2:.0f} MB, regardless of max_model_len.")

shared = Sequence(alloc, 300)
before = alloc.used
child = alloc.fork(shared.blocks)          # two sequences, one copy of the prompt KV
print(f"\nprefix sharing: forking a 300-token prompt for a second sequence allocated "
      f"{alloc.used - before} new blocks (copy-on-write).")
print("This is how vLLM serves n>1 sampling, beam search and shared system prompts cheaply,")
print("and it is the mechanism behind prefix caching (enabled by default in vLLM 0.29).")
print("\nvLLM facts read from the v0.29.0 source: cache_dtype default 'auto' (fp8 variants available),")
print("enable_prefix_caching default True, gpu_memory_utilization default 0.92, block_size auto-derived.")
print("When blocks run out the scheduler preempts sequences (recompute or swap) instead of crashing:")
print("that is why throughput degrades gracefully instead of OOM-ing under load.")
