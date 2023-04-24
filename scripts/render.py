from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeoMipTerrain, DirectionalLight
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.showbase.Loader import Loader
from direct.showbase import DirectObject


    
class MyApp(ShowBase):

    # imports a texture with specified mode ("ADD" or "REPLACE" or "DECAL"))
    def importTexture(self, path:str, mode:str, scale:float):
        terrain_texture = loader.loadTexture(path)
        terrain_texture_stage = TextureStage("terrain_texture_stage")

        # checks which mode user used
        if mode == "REPLACE":
            terrain_texture_stage.setMode(TextureStage.MReplace)
        elif mode == "ADD":
            terrain_texture_stage.setMode(TextureStage.MAdd)
        elif mode == "DECAL":
            terrain_texture_stage.setMode(TextureStage.MDecal)
            
        self.terrain.getRoot().setTexture(terrain_texture_stage, terrain_texture)
        self.terrain.getRoot().setTexScale(terrain_texture_stage, scale, scale)

    # initiates the height map with specified image
    def initiateHeightMap(self, path:str, block_size: int, vertical_size: int):
        self.terrain = GeoMipTerrain("Maze")
        self.terrain.setBlockSize(block_size)
        self.terrain.getRoot().setSz(vertical_size) # vertical size
        self.terrain.setHeightfield(path)
        self.terrain.setBruteforce(True) # non-dynamic maze
        self.terrain.getRoot().reparentTo(render)
        self.terrain.generate()


    def __init__(self):
        ShowBase.__init__(self)
        self.terrain = None # initializes terrain to none
        self.initiateHeightMap("generated/maze_gray.png", 32, 10) # creates terrain
        self.importTexture("generated/maze.mp4", "REPLACE", 317/513) # imports video onto map

        


app = MyApp()
app.run()