# Smoke Test Policy

- Always attempt a local smoke test before cluster submission to reduce queue-time waste and failed jobs.
- Local environment should be treated as CPU-only for validation purposes.
- Smoke tests should prioritize:
  - startup/initialization correctness,
  - environment creation/wrapper wiring,
  - short-step execution without crashes.
- Keep local smoke tests lightweight; avoid computationally intensive long training loops that are unlikely to finish locally.
