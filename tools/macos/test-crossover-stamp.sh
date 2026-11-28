#!/bin/sh
# Verify setup's Wine cache key without a Wine build or download.
set -eu
builder="$(dirname "$0")/build-crossover-wine.sh"

first=$(CROSSOVER_VERSION=26.0.1 "$builder" --stamp)
second=$(CROSSOVER_VERSION=26.0.2 "$builder" --stamp)
again=$(CROSSOVER_VERSION=26.0.1 "$builder" --stamp)

if [ "$first" = "$second" ] || [ "$first" != "$again" ]; then
    echo "the CrossOver build stamp must identify the selected release" >&2
    exit 1
fi
