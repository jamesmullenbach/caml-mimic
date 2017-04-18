#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "usage: $0 in_file out_file num_records"
    exit 1
fi
head -n 1 $1 > $2
tail -n +2 $1 | shuf -n $3 >> $2
python sort_by_length.py $2 1
