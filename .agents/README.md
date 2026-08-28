# Shared agent harness

This directory is the source of truth for project-level AI harness files.

- `PROJECT.md` contains repository instructions shared by Claude Code and Codex.
- `skills/` contains skills in the common `SKILL.md` format.
- `agents/*.md` contains canonical Claude-style subagent prompts.
- `agents/*.toml` is generated from the matching Markdown file for Codex.
- `settings.json`, `hooks/`, and `scripts/` contain Claude-specific configuration
  and utilities that still benefit from living beside the shared files.

The repository-root `.claude` and `.codex` directories contain only relative
symlinks into this directory. `CLAUDE.md` and `AGENTS.md` both link to
`PROJECT.md`.

After editing an agent Markdown file, regenerate and validate the Codex shims:

```bash
python3 .agents/scripts/sync_agent_shims.py
```

The generator intentionally maps only the portable fields: `name`,
`description`, and the Markdown body (as Codex `developer_instructions`). Claude
tool allowlists remain in the Markdown frontmatter because Codex controls tools
through its own runtime policy.
