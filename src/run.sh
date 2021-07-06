#!/bin/bash
for i in {0..25}
do
  #python3 test.py --filename="$i" --numofgames="60" |& tee "$i".out
  python3 test.py --height=20 --width=20 --filename="$i" --numofgames="25" |& tee logs_train_test/"$i".out
  # python3 test.py --height=20 --width=20 --filename="$i" --numofgames="25"
  #python3 test.py --height=20 --width=20 --filename="$i" --numofgames="25"
done
