#!/bin/bash

python3 main.py \
--horizon 10000 \
--failure_level 0.05 \
--dimension 5 \
--number_arms 100 \
--seed 8129 \
--env Logistic \
--desired_norm 2

