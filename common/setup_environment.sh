#!/bin/bash

module load craype-accel-amd-gfx908
module load rocm

echo "Currently loaded modules:"
echo "-------------------------"
module -t list
