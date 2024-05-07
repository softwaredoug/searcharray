#!/bin/bash


TEST="$*"

get_latest_file() {
    local directory=$1
    local extension=$2
    # Find the most recently updated file with the specified extension in the directory
		find "$directory" -type f -name "*.$extension" -exec stat -f "%m %N" {} + | sort -n -r | head -n 1 | cut -d' ' -f2-
}

python -m pytest -s -x --benchmark-disable "$TEST"
snakeviz "$(get_latest_file ".benchmarks" "prof")"
