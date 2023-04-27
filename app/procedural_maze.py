# type: ignore
# Extended from https://github.com/panda3d/panda3d/blob/master/samples/procedural-cube/main.py
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import numpy as np
from direct.task.Task import Task
from direct.actor.Actor import Actor



def make_grid(width, height, depth, texture_size=None):
    if texture_size is None:
        texture_size = (width-1, height-1)
    format = GeomVertexFormat.getV3n3cpt2()
    vdata = GeomVertexData('grid', format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    color = GeomVertexWriter(vdata, 'color')
    texcoord = GeomVertexWriter(vdata, 'texcoord')
    
    for y in range(height):
        for x in range(width):
            for z in range(depth):
                vertex.addData3f(x, y, z)
                normal.addData3f(0, 0, 1)
                color.addData4f(1, 1, 1, 1)
                texcoord.addData2f(x/texture_size[0], 1 - y/texture_size[1]-1)
    
    return vdata

def build_maze(maze, maze_gray, texture_size=None):
    vdata = make_grid(maze.shape[1] + 1, maze.shape[0] + 1, 2, texture_size)
    
    def idx(x, y, z):
        return z + x * 2 + y * 2 * (maze.shape[1] + 1)
    
    tris = GeomTriangles(Geom.UHStatic)
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            is_wall = (maze_gray[y, x] != 0)
            tris.addVertices(
                idx(x       , y     , int(is_wall)),
                idx(x + 1   , y + 1 , int(is_wall)),
                idx(x + 1   , y     , int(is_wall)),
            )
            tris.addVertices(
                idx(x       , y     , int(is_wall)),
                idx(x       , y + 1 , int(is_wall)),
                idx(x + 1   , y + 1 , int(is_wall)),
            )
            
            if is_wall:
                # Check adjacent cells
                if x == 0 or maze_gray[y, x - 1] == 0:
                    tris.addVertices(
                        idx(x       , y + 1 , 1),
                        idx(x       , y + 1 , 0),
                        idx(x       , y     , 0),
                    )
                    tris.addVertices(
                        idx(x       , y     , 1),
                        idx(x       , y + 1 , 1),
                        idx(x       , y     , 0),
                    )
                if x == maze.shape[1] - 1 or maze_gray[y, x + 1] == 0:
                    tris.addVertices(
                        idx(x + 1   , y     , 0),
                        idx(x + 1   , y + 1 , 0),
                        idx(x + 1   , y + 1 , 1),
                    )
                    tris.addVertices(
                        idx(x + 1   , y     , 0),
                        idx(x + 1   , y + 1 , 1),
                        idx(x + 1   , y     , 1),
                    )
                if y == 0 or maze_gray[y - 1, x] == 0:
                    tris.addVertices(
                        idx(x + 1   , y     , 1),
                        idx(x + 1   , y     , 0),
                        idx(x       , y     , 0),
                    )
                    tris.addVertices(
                        idx(x       , y     , 1),
                        idx(x + 1   , y     , 1),
                        idx(x       , y     , 0),
                    )
                if y == maze.shape[0] - 1 or maze_gray[y + 1, x] == 0:
                    tris.addVertices(
                        idx(x       , y + 1 , 0),
                        idx(x + 1   , y + 1 , 0),
                        idx(x + 1   , y + 1 , 1),
                    )
                    tris.addVertices(
                        idx(x       , y + 1 , 0),
                        idx(x + 1   , y + 1 , 1),
                        idx(x       , y + 1 , 1),
                    )
                
    grid = Geom(vdata)
    grid.addPrimitive(tris)
    snode = GeomNode('maze')
    snode.addGeom(grid)
    
    return snode