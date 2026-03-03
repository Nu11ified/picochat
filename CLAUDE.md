# Picochat — Project Guidelines

## Build & Test

- **Build**: `PATH="/home/nullify/.cargo/bin:$PATH" cargo build --workspace`
- **Test**: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test --workspace`
- **Test one crate**: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-core`
- **Test one file**: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-core --test kv_cache_test`

## Comment Style

- Add comments to explain **why**, not **what** — the code should speak for itself on the what.
- Comment complex algorithms, non-obvious invariants, and subtle correctness constraints (e.g., mask alignment, tensor shape requirements, numerical stability tricks).
- Comment magic numbers, non-trivial constants, and formulas with their derivation or source.
- Do **not** comment self-explanatory code: simple assignments, standard patterns, obvious control flow.
- Do **not** write meta-comments about what you (the agent) are currently doing (e.g., "// Now we add the loss function"). Comments describe the code's purpose, not the development process.
- Do **not** add section divider comments (e.g., `// --- Initialization ---`) unless the function exceeds ~50 lines and the sections represent meaningfully distinct phases.
- Prefer doc comments (`///`) on public items. Keep them to one sentence unless the API has non-obvious behavior.

## Code Rules

- No TODOs, placeholders, or stub code. Every implementation must be complete.
- No unnecessary abstractions. Three similar lines > premature helper function.
- DRY, YAGNI, TDD.
