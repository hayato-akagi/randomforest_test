#!/bin/bash

set -e

CONTAINER_TOOL=podman

echo "🛠️ Building containers..."
$CONTAINER_TOOL compose build --no-cache

echo "🚀 Running Compose..."
$CONTAINER_TOOL compose up
