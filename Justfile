default:
    @just --list

# Build the workspace
build:
    cargo build

# Run all tests
test:
    cargo nextest run

# Run clippy lints
lint:
    cargo clippy --all-targets -- --deny warnings

# Format code
fmt:
    cargo fmt
    taplo fmt

# Check formatting without modifying
fmt-check:
    cargo fmt -- --check
    taplo fmt --check

# Run all checks (lint + test + fmt)
check: fmt-check lint test

# Build documentation
doc:
    cargo doc --no-deps

# Run the CLI
run *ARGS:
    cargo run -p bikipy-cli -- {{ARGS}}

# Audit dependencies
audit:
    cargo audit
    cargo deny check
