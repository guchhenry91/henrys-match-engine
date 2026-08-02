#!/usr/bin/env bash
# Install the repo's git hooks into this clone. Hooks live in scripts/hooks/ so
# they are version-controlled; .git/hooks/ is not, so each clone must install
# once. Safe to re-run.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
src="$repo_root/scripts/hooks/pre-commit"
dst="$repo_root/.git/hooks/pre-commit"

cp "$src" "$dst"
chmod +x "$dst"
echo "installed pre-commit hook -> $dst"
