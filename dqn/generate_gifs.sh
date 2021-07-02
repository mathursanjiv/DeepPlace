#!/bin/bash
for i in {0..24}
do
  python3 generate_gifs.py --filename="$i" --numofgames="25"
done
