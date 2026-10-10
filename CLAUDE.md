@AGENTS.md

# Claude Code specific notes

- The rules in AGENTS.md above are the single source of truth. Do not duplicate or override them here.
  Add a rule to AGENTS.md so every assistant follows it.
- Read `ARCHITECTURE.md` before starting any build step, and work one step of its §10 build order at a time.
- Before running any script that may call a paid API, show the estimated number of calls and wait for
  confirmation.
- Never open or read `.env`. If a key is needed, refer to the variable name from `.env.example`.
- Prefer showing a diff-sized change over rewriting whole files.
