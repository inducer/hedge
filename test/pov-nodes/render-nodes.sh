#! /bin/sh

set -e
python nodal_points.py
#povray +Inodes.pov +Onodes.png +w320 +h200 +a -d
povray +Inodes.pov +Onodes.png +w1600 +h1200 +a -d
eog nodes.png
