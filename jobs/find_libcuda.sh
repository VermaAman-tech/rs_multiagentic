#!/bin/bash
#SBATCH --job-name=find-libcuda
#SBATCH --output=logs/find_libcuda_%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
find / -name "libcuda.so.1" -type f 2>/dev/null | head -20
find / -name "libcuda.so" -type f 2>/dev/null | head -10
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
/sbin/ldconfig -p 2>/dev/null | grep libcuda | head -10
