#!/bin/sh
# Streams the 12 demo clips through Apple's on-device dictation in real time (~1.5 min), then uploads the log.
# Needs macOS 26 and the Swift toolchain (xcode-select --install).
set -e
cd "$(dirname "$0")"
swift apple_dictation.swift manifest.jsonl > apple-dictation.jsonl
aws s3 cp apple-dictation.jsonl s3://$EARLYBIRD_BUCKET/tmp/apple/apple-dictation.jsonl --only-show-errors 2>/dev/null || curl -sf -T apple-dictation.jsonl "$PUT_URL" ; echo "uploaded $(wc -l < apple-dictation.jsonl) clips, thanks"
