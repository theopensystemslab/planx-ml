#!/usr/bin/env bash
# run like `./sync.sh host` on local, to sync data/ directory with remote

if [[ -z "${1-}" ]]; then
    REMOTE="g6"
else
    REMOTE="$1"
fi

rsync -avz data/ "$REMOTE:planx-ml/data/"
