#!/bin/bash

for i in {0..25}
do

echo $i
grep OptimalWL logs_train_test/$i.out | tail -1
grep OneShot logs_train_test/$i.out
grep "Best WL" logs_train_test/$i.out  | tail -1
done
