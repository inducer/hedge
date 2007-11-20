#! /bin/sh
# original soliday: 
# qsub -V -cwd -j y -pe openmpi 10 launch.sh
qsub -cwd -j y -pe openmpi 10 launch.sh
