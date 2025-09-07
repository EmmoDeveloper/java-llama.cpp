#!/bin/bash
# Usage: ./latest_git_status.sh [directory]
# If no directory is provided, uses current directory

DIR="${1:-.}"
cd "$DIR" || exit 1

git --no-pager log -1 --pretty=format:"🔹 Latest Commit:%n  Hash: %H%n  Short Hash: %h%n  Author: %an%n  Date: %ad%n  Message: %s%n%n🔹 Branch Info:%n  Current Branch: $(git branch --show-current)%n  Upstream: $(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || echo 'none')%n%n🔹 Tag Info:%n  Nearest Tag: $(git describe --tags --abbrev=0 2>/dev/null || echo 'none')%n%n🔹 Status:%n$(git status --short)%n%n🔹 Contributors:%n$(git shortlog -s -n | head -10)%n%n🔹 File Tree Snapshot (first 20 files):%n$(git ls-tree -r --name-only HEAD | head -20)"
