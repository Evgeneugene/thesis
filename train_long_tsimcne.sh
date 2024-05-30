#!/bin/bash

# Run the first script and wait for it to complete
python train_tsimcne_long_blood.py

# Run the second script and wait for it to complete
python train_tsimcne_long_dermamnist.py

# Run the third script and wait for it to complete
python train_tsimcne_long_leukemia.py
