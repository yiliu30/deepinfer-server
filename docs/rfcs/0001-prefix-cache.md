# RFC 0001: Prefix Cache Support

- Status: Proposed
- Date: 2026-10-21
- Author: Yi Liu
- Repository: deepinfer-server

## Summary

`deepinfer-server` currently re-prefills the full prompt for every request.
This is correct but expensive for multi-turn clients such as Claude Code,
where most of each turn is an unchanged conversation prefix.

This RFC proposes an opt-in prefix cache with two phases:

1. A server-side exact-prefix cache that retains completed DeepInfer
   sequences and reuses them through `Scheduler.fork_seq`.
2. An optional engine-level content-addressed cache in DeepInfer if the
   server-side cache is insufficient.

Phase 1 is small enough to ship in this repository. Phase 2 belongs upstream
in DeepInfer and should be treated as a separate project.

## Motivation

The gateway creates a new DeepInfer sequence for every Anthropic request. The
request is released after generation, so its KV blocks and state are freed.
Even when the next turn shares a long prefix, the engine starts from token
zero.

DeepInfer already provides an explicit prefix-reuse primitive:

```text
Scheduler.fork_seq(
    source_seq, blocks, target_branch_id, sampling_params,
    so_matcher, sp_token, replace_last_token_id, seq_id, arrival_time,
)
```

`fork_seq` pins the source sequence's KV blocks and state, clones the state,
and creates a new sequence that starts from the source's cached prefix. It is
used by DeepInfer's benchmark for parallel sampling. It is not an automatic
prompt-hash cache, and the gateway does not call it today.

The recipe layer already has the accounting hook we need:

```text
PromptUsage(prompt_tokens=..., prompt_cache_hit_tokens=...)
```

The gateway currently leaves `prompt_cache_hit_tokens` at zero.

## Goals

- Avoid re-prefilling unchanged conversation prefixes for repeated requests.
- Preserve the existing Anthropic Messages API and client behavior.
- Keep the feature opt-in and fall back to full prefill on any uncertainty.
- Report cache reads through Anthropic usage fields.
- Keep multi-rank EP/DP scheduling deterministic.
- Bound retained KV and state resources with explicit eviction.

## Non-goals

- Image or multimodal prefix caching in the first version.
- Cross-model or cross-tokenizer cache reuse.
- Automatic reuse of arbitrary partial prefixes in Phase 1.
- Changing the model's chat template or tokenizer.
- Making prefix caching mandatory for every deployment.

## Current Constraints

- The gateway has four DeepInfer ranks in the documented deployment, with
  expert parallelism and data parallelism both set to four.
- Each rank owns a separate scheduler and KV pool. A cached prefix exists only
  on the rank that produced it.
- A retained sequence consumes KV blocks and a state slot. State slots are
  `int(max_num_seqs * DEEPINFER_NUM_STATE_SLOTS_RATIO) + 1`; the default ratio
  is 1.2.
- `fork_seq` clones the source token sequence. Replacing the fork's token data
  after creation is suitable for an exact full-source prefix, but not for a
  shorter arbitrary prefix.
- Assistant output is detokenized by the client and retokenized on the next
  turn. The resulting token IDs may differ at the boundary even when the text
  is unchanged.
- DeepInfer's block manager is currently a free-list allocator, not a
  content-addressed prefix cache.

## Phase 1: Server-Side Exact-Prefix Reuse

### Overview

Retain a small LRU of completed sequences on each rank. For a new request,
choose a stable conversation key, route to the rank that owns the previous
sequence, and fork that sequence if the new prompt begins with its cached
tokens.

This reuses DeepInfer's existing block/state pinning and clone logic. It does
not require a new block allocator.

### Components And Changes

#### `server.py`

- Compute an affinity key from the first `GATEWAY_PREFIX_CACHE_KEY_TOKENS`
  prompt tokens, for example with BLAKE2.
- Pass the key and token IDs to `EnginePool.submit`.
- Wait for the rank's cache decision before emitting
  `InferenceChunk.ready`.
- Set `PromptUsage.prompt_cache_hit_tokens` to the actual matched count.
- Keep cache creation accounting consistent with the recipe's usage mapping.

#### `engine_pool.py`

- Extend the route record with the affinity key, source-token snapshot, and
  cache-hit metadata.
- Maintain `affinity_key -> rank` so subsequent turns use the same rank.
- On a hit, send `fork_source` to the rank instead of `add`.
- On a miss, send a normal `add` and record the completed sequence for future
  turns.
- Add an LRU eviction policy at the driver and send `drop_cache` to ranks.
- Keep the existing lockstep `step()` behavior; a fork is local scheduler
  state and must not change EP collective participation.

#### Rank Process In `engine_pool.py`

- Replace the default `LLMEngine.finalize_seq` behavior for cacheable
  requests with a retention hook. A retained sequence must not call
  `scheduler.free_seq` at completion.
- Store the retained `Sequence`, its token IDs, cached-token count, and
  last-used timestamp in an ordered dictionary.
- On eviction, call `scheduler.free_seq(sequence)` to release KV blocks and
  the state slot.
- On a fork request:
  - Verify the source still exists and has `num_cached_tokens > 0`.
  - Verify the new prompt is longer than the cached prefix.
  - Verify the new prompt's token prefix matches the source's token prefix.
  - Call `scheduler.fork_seq` with a branch ID beginning with `_`.
  - Replace the fork's `data` and `compressed_data` with the new full prompt.
  - Set the fork's `request_id`, `num_cached_tokens`, and
    `cached_hit_tokens`.
  - Report the actual matched token count to the driver.
- Fall back to a normal `add_request` on any validation failure.

The intended call shape is:

```python
fork = engine.scheduler.fork_seq(
    source_seq,
    None,
    f"_{request_id}",
    SamplingParams(**sampling_params),
    source_seq.so_matcher,
    source_seq.data.get_last_token_id(),
    -1,
    next(engine.seq_counter),
    time.monotonic(),
)
fork.data = SequenceData(prompt_token_ids)
fork.compressed_data = SequenceData(prompt_token_ids)
fork.request_id = request_id
fork.num_cached_tokens = source_seq.num_cached_tokens
fork.cached_hit_tokens = source_seq.num_cached_tokens
```

This shape depends on the current Python bindings. A cleaner upstream API is
described in Phase 2.

### Configuration

Initial proposed knobs:

```text
GATEWAY_PREFIX_CACHE=0
GATEWAY_PREFIX_CACHE_MAX_ENTRIES=1
GATEWAY_PREFIX_CACHE_MIN_TOKENS=4096
GATEWAY_PREFIX_CACHE_KEY_TOKENS=4096
```

`MAX_ENTRIES` is per rank. The default is conservative because each retained
entry consumes one state slot. Deployments that need more entries should
increase `DEEPINFER_NUM_STATE_SLOTS_RATIO` and verify memory headroom.

### Usage Accounting

On a hit, report:

```text
prompt_tokens = len(prompt_token_ids)
prompt_cache_hit_tokens = matched_prefix_tokens
```

The cache-creation portion is the remainder of the prompt. A miss reports
zero cache-read tokens. The gateway must never report a hit larger than the
number of tokens actually present in the retained source.

### Eviction And Capacity

- Evict by least-recently-used prefix per rank.
- Evict before retaining a new entry when the per-rank limit is reached.
- Never free a source while a fork is active; `fork_seq` pins the source
  blocks and state, and `scheduler.free_seq` only drops the source's own
  reference.
- Clear all retained sequences during shutdown.
- Expose retained-entry count, evictions, hits, misses, and retained tokens in
  `/health` or metrics.

### Failure Handling

Fall back to normal prefill when:

- The affinity rank has no retained entry.
- The retained sequence is shorter than or equal to the new prefix.
- The token prefix does not match exactly.
- The source has structured-output state or multimodal data.
- `fork_seq` raises any exception.

An aborted or failed request must not be retained.

### Security And Isolation

The first version should namespace cache keys by API key or user identity when
the deployment is multi-tenant. Sharing a cache across users can create
timing side channels and makes resource accounting ambiguous.

## Phase 2: Engine-Level Automatic Prefix Cache

If the server-side cache has a low hit rate or too many routing constraints,
move the cache into DeepInfer. This is the long-term, provider-agnostic
solution.

### `csrc/engine/block_manager.hpp`

Add a content-addressed block cache:

- Hash full prompt blocks, including the parent hash and token IDs.
- Map block hash to block ID.
- Track reference counts and LRU order.
- Pin matched blocks for a new sequence.
- Unpin and evict blocks when their reference count reaches zero.

### `csrc/engine/scheduler.hpp`

- In `_schedule_prefill`, compute hashes for full prompt blocks and find the
  longest cached prefix before allocating new blocks.
- Set the sequence's block table, `cache.num_tokens`, and `cache.hit_tokens`
  for the matched prefix.
- Allocate and prefill only the remaining suffix.
- Insert newly completed full blocks into the cache after they are computed.
- Update `free_seq_nolock` to decrement cache references rather than returning
  blocks directly to the free list.

The existing `prefix_caching_hit` flag around scheduler line 373 is not an
automatic cache lookup; it is part of the explicit fork/chunked-prefill path.

### State And Model Constraints

DeepSeek-V4.1 has compressed and index attention caches plus per-sequence
state. KV blocks alone are not enough for safe prefix reuse. A cache entry
must also pin the state snapshot, or the scheduler must clone the state into a
fresh state slot using the same mechanism as `fork_seq`.

This is the main reason Phase 2 is larger than Phase 1. It needs tests for
sliding-window state, compressed attention, chunked prefill, preemption, and
all supported model configurations.

### Python And API Changes

- Propagate `cached_hit_tokens` through `RequestOutput` and the recipe usage
  path.
- Expose any new sequence or block-manager methods through
  `csrc/engine/pybind_sequence.cpp` and `deepinfer/_C/engine.pyi`.
- Remove the need for the gateway to call `fork_seq` directly.

## Correctness Invariants

- A cache hit is valid only when the token prefix and tokenizer match.
- No cached block or state may be freed while any fork references it.
- A cache miss must produce exactly the same model output as no cache.
- Cache state must not cross model revisions or tokenizer configurations.
- Multi-rank scheduling must remain lockstep; a fork on one rank must not
  alter the collective participation of other ranks.
- Cached sequences must never be reused for multimodal or structured-output
  requests in Phase 1.

## Testing Plan

Unit tests:

- Prefix-key hashing and collision handling.
- Exact-prefix validation and fallback.
- LRU eviction and reference-count release.
- Usage accounting for hit, miss, and abort.

Integration tests:

- Two-turn Claude Code conversation with a cache hit.
- A prompt above the old 65K boundary.
- A 1M-context request with a retained prefix.
- Multiple concurrent sessions on different ranks.
- Forced eviction while a fork is active.
- Aborted request followed by a new request.
- Tokenization mismatch fallback.
- Structured-output and multimodal requests bypassing the cache.
- `/health` and metrics accuracy.

Performance tests:

- TTFT and prefill-token reduction for repeated 90K and 1M-token prompts.
- No regression on total throughput for cache misses.
- Memory growth over a long-running multi-session workload.

## Rollout

1. Ship Phase 1 disabled by default.
2. Enable it in a single-user deployment with one retained entry per rank.
3. Compare TTFT, hit rate, state-slot pressure, and memory.
4. Increase entries only after confirming state-slot and KV headroom.
5. Decide whether Phase 2 is justified by the measured hit rate.

The gateway must be able to disable the cache without changing the API or
restarting clients.

## Alternatives Considered

- **Require `cache_control` breakpoints from clients.** The gateway currently
  does not interpret them, and relying on clients would not help clients that
  do not send them.
- **Use a single DP rank.** This simplifies cache locality but loses the
  existing four-way data parallelism.
- **Implement only the engine-level block cache.** It is the cleanest long-term
  design, but the state-cloning and block-hash work is much larger than Phase 1.
- **Cache prompt-only state instead of completed sequences.** This could
  improve reuse when assistant output retokenizes differently, but requires a
  new sequence snapshot API in DeepInfer.

## Open Questions

- Should the cache key include the API key, conversation ID, or only the token
  prefix?
- Should Phase 1 cache the completed sequence or a prompt-only snapshot?
- How should `cache_control` breakpoints map to token offsets if the recipe
  starts preserving them?
- What is the right default number of retained entries per rank?
- Should eviction be LRU by entry count, by retained tokens, or by state-slot
  pressure?
- Can `fork_seq` safely accept a new prompt directly, or should DeepInfer add
  `fork_seq_with_tokens` instead of assigning `Sequence.data` after the fact?
- Is the state-slot ratio increase safe across all supported GPU and model
  configurations?

## References

- DeepInfer `Scheduler.fork_seq` and `fork_seq` benchmark usage.
- DeepInfer `BlockManager` and `Scheduler._schedule_prefill`.
- `deepseek-recipe` `PromptUsage` cache-hit accounting.
- Anthropic Messages API cache usage fields.
