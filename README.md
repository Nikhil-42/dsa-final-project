# dsa-final-project
Our DSA-Final-Project is a 3D path finding algorithm visualizer.  The maze is 317 x 317 pixels and is traversed with the following:
* Breadth First Search
* Dijkstra's Algorithm
* A* Search Algorithm
* Depth First Search

Each algorithm starts in a seperate corner, each with a corresponding color.  The different colors on the map represent different terrain types that hinder player movement  according to the intensity of the color.  The time taken for each algorithm to reach the center of the maze is recorded, when the center is reached a black line tracing the shortest path is displayed.  The maze is stored as an adjacency list with edge weights determined by walls and terrain types.

<p align="center">
  <img src="https://user-images.githubusercontent.com/84941950/234443340-c45f0db0-dc1b-4b1d-bf11-9039706fcce8.gif" alt="animated" 
    width="500"
    height="500"/>
</p>

##  Tools Used
* [**FFMPEG**](https://github.com/FFmpeg/FFmpeg): Visualizes algorithms on a 2D maze and outputs a video
* [**Panda3D**](https://github.com/panda3d/panda3d): Used to render our heightmap into three-dimensions
* [**Numba**](https://github.com/numba/numba): A JIT compiler to speed up maze generation
* [**Pillow**](https://github.com/python-pillow/Pillow): Saves the the maze heightmap as a png
* [**Numpy**](https://github.com/numpy/numpy): Python library for better arrays

## Build
The required dependencies are stored in a requirements.txt file which can be installed with the following:

``` python -m pip install -r requirements.txt ```

The following commands should then be ran in the terminal:

1. To generate the maze ```python maze-generation/generate_maze.py ```

2. To output a 2D video of the algorithms ```python scripts/run_search.py```

3. To render the maze in 3D with Panda3D ```python scripts/render.py```

## Notes
The program uses a 2D maze as a png that looks like the following:
<p align="center">
  <img src="https://user-images.githubusercontent.com/84941950/234447848-097579ff-ed55-4668-88dc-6eb24e46d2a7.png" 
   width="500"
   height="500"/>
 </p>
 
The render program will then use a heightmap to transform that png into a 3D terrain.  When the run_searches program is ran, it will add all color to the map and output a video of each traversal.  Furthermore, it will output the total weights of the path's taken and the average time it took each agent to reach the center.

<p align="center">
  <img src="https://user-images.githubusercontent.com/84941950/234444107-8efdc42f-1565-47b0-a393-45bd7e917074.gif" 
   width="500"
   height="500"/>
 </p>




