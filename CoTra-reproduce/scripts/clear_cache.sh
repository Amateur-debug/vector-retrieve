#!/bin/bash

# sudo ../scripts/clear_cache.sh
echo 3 | sudo tee /proc/sys/vm/drop_caches
