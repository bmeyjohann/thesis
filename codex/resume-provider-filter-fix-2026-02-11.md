# Resume History Provider-Filter Fix (2026-02-11)

## Root Cause

`codex resume` picker filters sessions by the current `model_provider` id. Mixed historical session metadata (`openai` vs `codex-lb`) caused valid sessions to be hidden from the picker.

## Applied Fix

- Created full backups:
  - `/tmp/codex-backups/provider-fix-20260211-174815/.codex.tgz`
  - `/tmp/codex-backups/provider-fix-20260211-174815/.codex-lb.tgz`
- Normalized existing session metadata to be provider-agnostic by removing `model_provider` from `session_meta` in all files under `~/.codex/sessions`.
- Files changed: 73 session files.

## Verification

- Target session `019c2a7a-9067-7a13-925e-ef9b5c306584` now has no `model_provider` in `session_meta`.
- Simulated picker visibility for cwd `/home/benjamin/thesis`:
  - Current provider `codex-lb`: target visible
  - Current provider `openai`: target visible

## Durable Follow-Up (Applied)

- Updated `~/.codex/config.toml` default provider:
  - `model_provider = "openai"`
- Kept `[model_providers.codex-lb]` block for explicit fallback if needed.

Run Codex through codex-lb while keeping provider identity unified (`openai`):

```bash
OPENAI_BASE_URL=http://127.0.0.1:2455/backend-api/codex codex
```

Run direct OpenAI (no load balancer):

```bash
codex
```

This keeps resume history under a single provider id and avoids hidden sessions when switching routes.

## Rollback

1. `rm -rf /home/benjamin/.codex /home/benjamin/.codex-lb`
2. `tar -xzf /tmp/codex-backups/provider-fix-20260211-174815/.codex.tgz -C /home/benjamin`
3. `tar -xzf /tmp/codex-backups/provider-fix-20260211-174815/.codex-lb.tgz -C /home/benjamin`
