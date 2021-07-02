#!/bin/bash
for i in {1..100}
do
  cp train_data/min"$i".pkl train_test_data/block_$i.pkl
done
