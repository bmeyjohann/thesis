# Corrections Log

- 2026-02-11: Resume-history debugging initially focused too much on cache/state reset. Correct approach: first check provider filtering (`config.model_provider` vs each session's `session_meta.model_provider`) before doing resets, because `codex resume` lists by current provider.
