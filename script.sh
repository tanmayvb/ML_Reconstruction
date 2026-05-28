#!/bin/bash

git rev-list --objects --all | \
git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize:disk) %(rest)' | \
grep '^blob' | \
sort -k3 -n -r | \
head -20 | \
awk '{ printf "%.2f MB\t%s\n", $3/1024/1024, $4 }'
