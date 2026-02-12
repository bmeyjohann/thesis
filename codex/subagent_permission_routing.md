# Subagent Permission Routing (2026-02-11)

- Trigger: user reported subagent permission/approval flow issues.
- Policy: subagents must not run escalated commands.
- Routing rule: subagents collect findings and required privileged commands; main agent performs escalated operations after explicit user approval.
- Scope includes SSH/cluster access and any command that prompts for permission.
