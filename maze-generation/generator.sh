#!/bin/bash

for (( i=1; i<=100; i++ ))
do
  python3 iterative.py > mazes/mazes$i.txt
done
