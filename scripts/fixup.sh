#!/bin/bash

# Specify the glob pattern for the files you want to process
# Example: '*.txt' for all text files

# Loop over all files matching the specified pattern
for file in "$@"
do
    # Check if the file exists and is a regular file
    if [ -f "$file" ]; then
        # Use sed to remove trailing spaces and save changes back to the file
        contents_before="$(cat "$file")"
        sed -i '' 's/[[:blank:]]*$//' "$file"
        contents_after="$(cat "$file")"
        if [ "$contents_before" != "$contents_after" ]; then
            echo "$file was modified."
        fi
    fi
done
