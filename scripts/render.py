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
        terrain.setHeightfield("../generated/maze_gray.png")
        terrain.setBruteforce(True) # non-dynamic maze
        terrain.getRoot().reparentTo(render)
        terrain.generate()

        # loads colored terrain into a texture
        terrain_texture = loader.loadTexture("../generated/maze.png")
        terrain_texture_stage = TextureStage("terrain_texture_stage")
        terrain_texture_stage.setMode(TextureStage.MReplace)

        # sets the colored terrain to the map
        terrain.getRoot().setTexture(terrain_texture_stage, terrain_texture)
        terrain.getRoot().setTexScale(terrain_texture_stage, 1, 1)

        
        camera.setPos(-1000, 0, 0)


app = MyApp()
app.run()