import bpy
import random

WIDTH = 10
HEIGHT = 10

maze = [[random.randint(0,2) for _ in range(WIDTH)] for _ in range(HEIGHT)]

for x in range(len(maze)):
    for y in range(len(maze)):
        if maze[y][x] == 1:
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.object
            cube.scale = (0.5, 0.5, 0.5)
            cube.location = (x - 0.5 * WIDTH, y - 0.5 * HEIGHT, 1)
