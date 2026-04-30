#!/usr/bin/env bash
# Downloads and compiles the external Rust UTTT solver (nelhage/ultimattt).
# Requires: git, cargo (Rust toolchain)

set -euo pipefail

REPO_URL="https://github.com/nelhage/ultimattt"
TARGET_DIR="$(dirname "$0")/../external/ultimattt"

if [ -d "$TARGET_DIR/.git" ]; then
    echo "Solver repo already cloned at $TARGET_DIR"
else
    echo "Cloning $REPO_URL into $TARGET_DIR..."
    git clone --depth 1 "$REPO_URL" "$TARGET_DIR"
fi

echo "Building solver (release mode)..."
cargo build --release --manifest-path "$TARGET_DIR/Cargo.toml"

echo "Done. Binary at: $TARGET_DIR/target/release/ultimattt"
