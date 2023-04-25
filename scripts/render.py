from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import numpy as np

class MyApp(ShowBase):
    
    def setupCameraControls(self, props: WindowProperties):
        # create reasonable camera controls
        self.disableMouse()
        
        # Get the current keyboard layout.
        # This may be a somewhat expensive operation, so don't call
        # it all the time, instead storing the result when possible.
        keyboard_map = self.win.get_keyboard_map()

        # Find out which virtual keys are associated with the ANSI US "wasdqe"
        w_button = keyboard_map.get_mapped_button("w")
        a_button = keyboard_map.get_mapped_button("a")
        s_button = keyboard_map.get_mapped_button("s")
        d_button = keyboard_map.get_mapped_button("d")
        q_button = keyboard_map.get_mapped_button("q")
        e_button = keyboard_map.get_mapped_button("e")
        
        SPEED = 200
        def move_camera(direction):
            self.camera.setPos(self.camera.getPos() + render.getRelativeVector(camera, tuple(np.array(direction) * SPEED * globalClock.getDt())))
        
        self.accept(f"{w_button}-repeat", lambda: move_camera((0, 1, 0)))
        self.accept(f"{a_button}-repeat", lambda: move_camera((-1, 0, 0)))
        self.accept(f"{s_button}-repeat", lambda: move_camera((0, -1, 0)))
        self.accept(f"{d_button}-repeat", lambda: move_camera((1, 0, 0)))
        self.accept(f"{q_button}-repeat", lambda: move_camera((0, 0, -1)))
        self.accept(f"{e_button}-repeat", lambda: move_camera((0, 0, 1)))

        def rotate_camera(task):
            if self.mouseWatcherNode.hasMouse():
                x = self.mouseWatcherNode.getMouseX()
                y = self.mouseWatcherNode.getMouseY()
                self.camera.setH(self.camera.getH() - 100 * x)
                self.camera.setP(self.camera.getP() + 100 * y)
                self.win.movePointer(0, props.getXSize() // 2, props.getYSize() // 2)
            return task.cont
        
        self.taskMgr.add(rotate_camera, "rotateCamera")
        

    # imports a texture with specified mode ("ADD" or "REPLACE" or "DECAL"))
    def importTexture(self, path: str, mode: str, scale: float):
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
    def initiateHeightMap(self, path: str, block_size: int, vertical_size: int):
        self.terrain = GeoMipTerrain("Maze")
        self.terrain.setBlockSize(block_size)
        self.terrain.getRoot().setSz(vertical_size)  # vertical size
        self.terrain.setHeightfield(path)
        self.terrain.setBruteforce(True)  # non-dynamic maze
        self.terrain.getRoot().reparentTo(render)
        self.terrain.generate()

    def addText(self, text_str: str, coords: tuple, color: tuple):
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
        
        # exit on escape
        self.accept("escape", self.userExit)
        

        props = WindowProperties()
        
        # set relative mode and hide the cursor
        props.setCursorHidden(True)
        # sets window size
        props.setSize(1280, 720)
        
        self.setupCameraControls(props)
        
        self.win.requestProperties(props)

        # coords for the card
        left, right, top, bottom = -1.7, -.3, .95, .5

        # making the card
        cm = CardMaker('card')
        cm.setColor(105, 105, 105, 255)
        cm.setFrame(left, right, top, bottom)
        card = render2d.attachNewNode(cm.generate())

        self.addText("A*: ", (-1.7, 0, .55), (0, 255, 0, 1))
        self.addText("Depth First Search:", (-1.7, 0, .75), (255, 0, 255, 1))
        self.addText("Breadth First Search: ", (-1.7, 0, .65), (255, 0, 0, 1))
        self.addText("Dijkstra's: ", (-1.7, 0, .85), (0, 0, 255, 1))

        self.terrain = None  # initializes terrain to none
        self.initiateHeightMap("generated/maze_gray.png", 32, 20)  # creates terrain
        self.importTexture("generated/maze.mp4", "REPLACE", 317/513)  # imports video onto map


app = MyApp()
app.run()
