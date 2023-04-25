from direct.showbase.ShowBase import ShowBase
from panda3d.core import *


    
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
    
    def addText(self, text_str:str, coords: tuple, color:tuple):
        text = TextNode('node name')
        text.setText(text_str)
        text.setTextColor(color)
        textNodePath = aspect2d.attachNewNode(text)
        textNodePath.setScale(0.07)
        textNodePath.setPos(coords)
        cmr12 = loader.loadFont('cmr12.egg')
        text.setFont(cmr12)


    def __init__(self):
        ShowBase.__init__(self)
        # sets window size
        props = WindowProperties() 
        props.setSize(1280, 720) 

        left, right, top, bottom = -1.7, -.3, .95, .5

        cm = CardMaker('card')
        cm.setColor(105, 105, 105, 255)
        cm.setFrame(left, right, top, bottom)
        card = render2d.attachNewNode(cm.generate())


        self.addText("A*: ", (-1.7, 0, .55), (0, 255, 0, 1))
        self.addText("Bellman Ford: ", (-1.7, 0, .75), (255, 0, 255, 1))
        self.addText("Breadth First Search: ", (-1.7, 0, .65), (255, 0, 0, 1))
        self.addText("Dijkstra's: ", (-1.7, 0, .85), (0, 0, 255, 1))

        


        self.win.requestProperties(props) 
        self.terrain = None # initializes terrain to none
        self.initiateHeightMap("generated/maze_gray.png", 32, 20) # creates terrain
        self.importTexture("generated/maze.mp4", "REPLACE", 317/513) # imports video onto map

app = MyApp()
app.run()