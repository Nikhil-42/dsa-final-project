from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeoMipTerrain, DirectionalLight
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.showbase.Loader import Loader
from direct.showbase import DirectObject


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # uses a height map to generate the terrain from a b/w image
        terrain = GeoMipTerrain("Maze")
        terrain.setBlockSize(32)
        terrain.getRoot().setSz(10) # vertical size
        terrain.setHeightfield("generated/maze_gray.png")
        terrain.setBruteforce(True) # non-dynamic maze
        terrain.getRoot().reparentTo(render)
        terrain.generate()

        # loads colored terrain into a texture
        video_texture = loader.loadTexture("generated/maze.mp4")
        video_texture_stage = TextureStage("video_texture_stage")
        video_texture_stage.setMode(TextureStage.MReplace)

        # sets the colored terrain to the map
        terrain.getRoot().setTexture(video_texture_stage, video_texture)
        terrain.getRoot().setTexScale(video_texture_stage, 317/513, 317/513)

        


app = MyApp()
app.run()