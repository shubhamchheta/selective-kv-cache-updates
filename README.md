# Selective KV-Cache Updates for Textbook Knowledge

This project implements and evaluates a selective Key-Value (KV) cache
update strategy for large language models when a previously cached
knowledge document is modified.

The main idea is simple: when only one part of a document changes,
there is no need to recompute the complete KV cache. We reuse the
cached prefix before the first changed segment and recompute only the
changed segment and its causally dependent suffix.

This project was developed as part of my Master's thesis at the
University of Bayreuth.

---

## Why this project?

Large language models can use KV caching to avoid repeatedly
processing the same context during inference. However, when a
knowledge source changes, a conventional approach is to rebuild the
complete cache.

For a long document with a small modification, this can result in
unnecessary computation.

The challenge is that Transformer representations are causally
dependent on previous tokens. Therefore, simply recomputing the
changed chapter is not sufficient.

This project investigates how much of the existing cache can be
safely reused while still producing the correct updated cache.

---

## Approach

The document is divided into segments (chapters in the current
implementation).

When a segment changes:

1. Identify the first changed segment.
2. Reuse the KV cache corresponding to the unchanged prefix.
3. Recompute the changed segment together with all following segments.
4. Combine the reused prefix cache and the newly computed tail.
5. Use the updated cache for question answering.

### Overview

```text
Original document

[C1] [C2] [C3] [C4] [C5] [C6] [C7] [C8]
                     ↑
                 modification


Selective update

[C1] [C2] [C3] | [C4'] [C5] [C6] [C7] [C8]
      reused            recomputed
       prefix              tail
