# Colloid Sim 3D
![The simulator in action!](https://cdn.discordapp.com/attachments/560953599081840651/1079740957303521341/ezgif-5-6917e83500.gif)

## Setup
The only two packages you need are `numpy` and `moderngl`. So go ahead:

```
$ python3 -m pip install numpy moderngl
```

Then you should be able to run the main script:

```
$ python3 main.py
```

## Implemented Features
- [x] Collision between balls and box
- [x] Collision between balls and each other
- [x] Arbitrary system of balls in box
- [] Brownian motion
- [] Glass transition simulation
- [x] Hardware accelerated 3D graphcis
- [] High speed simulation using Quad/Oct/KD-trees for broad/narrow phase
- [] Arbitrary / non-spherical shape collision handling
- [] Arbitrary / non-spherical shape visualization

## Code organization
The main files for thie project are the following:
- `main.py`: Main executable. Contains some basic settings configuration. By default all particles are spawned with random position and velocities
- `ColloidSim.py`: Colloid simulation logic. Purlely numerical: when finished contains matrices of positions and velocities of particles over time, stored as 3D matrices.
- `ColloidViz.py`: Colloid simulation result visualizer. Contains zero simulation logic and is just for visualizing finished simulations. Handles all 3D rendering lifecycles.

Other helper files include:
- `moderngl_sphere.py`: Simple 3D animation script for validating that you have a working 3D rendering environment
- `graphics_utils.py`: Math functions for generating and manipulating shapes and cameras.
- `default.vert` and `default.frag`: Shaders used for 3D graphics. The vertex shader paraellizes vertex position calculations, while the fragment shader parallelizes color shading.