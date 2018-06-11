# Planar-Graph-for-Microscopic-image
Extracting a planar graph from a noisy microscopic image of a physical simulation

## Requirements:
  - OpenCV 3
  - Python 3
  - Boost Library
  
## Compilation:
  In Root directory with main.cpp run:
  ````
  $ g++ -std=c++11 main.cpp headers/edge_thinning.cpp headers/threshold.cpp -o planar.out `pkg-config --cflags --libs opencv` -lboost_system
  ````
  
## Run:

  ````
  $ ./planar.out
  ````
