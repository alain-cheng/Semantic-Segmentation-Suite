#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Commit message not provided"
    echo "Usage: ./git-commit-push.sh \"Your commit message\""
    exit 1
fi

git add .

git commit -m "$1"

git push

echo "Changes commit and pushed successfully!"