# agent_queue — how to answer pending prompts

The pipeline is paused (`WAITING_FOR_AGENT`, exit code 20). Each line of
`queue/requests.jsonl` is one prompt waiting for an answer:

    {"id": "<16 hex>", "kind": "generate|repair|extract", "arm": "P0|P2|S2|S2R|XL",
     "sample_id": "...", "system": "...", "user": "...", "schema": {...} | null}

Append ONE answer line per request to `queue/responses.jsonl`:

    {"id": "<same id>", "text": "<plain prose answer>"}          # prose modes (P0/P2)
    {"id": "<same id>", "json": { ... }}                        # structured modes (S2/S2R/extract)

Rules
- `text` and `json` are both accepted; include the one the request expects
  (`schema` non-null => structured answer; the object is validated against it).
- Do NOT edit `requests.jsonl` ids; ids are sha256-based and deterministic, so
  re-running the same command re-uses existing answers and caches.
- Then re-run the exact same command. When all pending ids are answered the
  pipeline continues to the next batch automatically.
