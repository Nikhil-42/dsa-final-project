# type: ignore
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import numpy as np
from direct.task.Task import Task
from direct.actor.Actor import Actor
import json
import cv2

class Maze(ShowBase):
    
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
        
        self.win.movePointer(0, props.getXSize() // 2, props.getYSize() // 2)
        self.taskMgr.add(rotate_camera, "rotateCamera")
        
    # imports a texture
    def importTexture(self, path: str):
        texture = loader.loadTexture(path)
        texture.setMagfilter(SamplerState.FT_nearest)
        texture.setMinfilter(SamplerState.FT_nearest)

        return texture

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
        
        
    async def follow_path(self, path, agent):
        self.amangi[agent].loop("run")
        for node in path:
            self.amangi[agent].setPos(node % 317 * 513/317 + 0.5, 513 - (node // 317 * 513/317 + 0.5), 1)
            await Task.pause(self.maze[node // 317, node % 317] / 255 / 100000)
        self.amangi[agent].stop()
        self.runner_finished[agent] = True
        return Task.done
    
    async def recurseSetup(self, rotation):
        # Swap video for image of solved maze
        self.video.stop()
        self.terrain.getRoot().setTexture(self.rotation_frames[0])
        self.terrain.getRoot().setTexScale(TextureStage.default, 1, 1)
        
        while not all(self.runner_finished.values()):
            await Task.pause(0.5)
        
        self.setupRunners((rotation + 1) % 4)
        return Task.done
        
    def setupRunners(self, rotation):
        # Display the video
        self.video.setTime(max((self.run_data[agent]['finish_timestamps'][rotation-1] for agent in self.run_data)))
        self.terrain.getRoot().setTexture(self.video)
        self.terrain.getRoot().setTexScale(TextureStage.default, 317/513, 317/513)
        self.video.play()
        
        for agent in self.run_data:
            node = self.run_data[agent]["paths"][rotation][0]
            self.runner_finished[agent] = False
            self.amangi[agent].setPos(node % 317 * 513/317 + 0.5, 513 - (node // 317 * 513/317 + 0.5), 1)
            self.amangi[agent].stop()
                
            self.taskMgr.doMethodLater(self.run_data[agent]['finish_timestamps'][rotation], self.follow_path, "FollowPath"+agent, extraArgs=[self.run_data[agent]["paths"][rotation], agent])
            
        self.taskMgr.doMethodLater(max((self.run_data[agent]['finish_timestamps'][rotation] for agent in self.run_data)), self.recurseSetup, "RecurseSetup", extraArgs=[rotation])

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
        

        self.terrain = None  # initializes terrain to none
        self.initiateHeightMap("generated/maze_gray.png", 32, 1)  # creates terrain
        self.video = self.importTexture("generated/maze.mpg")  # imports video onto map
        self.rotation_frames = []
        for rotation in range(4):
            self.rotation_frames.append(self.importTexture(f"generated/solved_maze_{rotation}.png"))
        
        # self.video.stop()
        self.terrain.getRoot().setTexture(self.video)
        self.terrain.getRoot().setTexScale(TextureStage.default, 317/513, 317/513)
        
        self.maze = cv2.imread("generated/maze_gray.png", cv2.IMREAD_GRAYSCALE)
        
        self.run_data = json.load(open("generated/output.json"))
        self.amangi = {}
        self.runner_finished = {}
        y = 0.55
        dy = 0.1
        for agent in self.run_data:
            self.addText(f"{agent}: {sum(self.run_data[agent]['times']) * 1000}", (-1.7, 0, y), tuple(self.run_data[agent]["color"][::-1] + [1]))
            y += dy

            new_amangus = Actor("models/amangus.glb")
            new_amangus.reparentTo(render)  # reparents amangus to render
            self.amangi[agent] = new_amangus

        self.setupRunners(0)

        self.camera.setPos((255, 255, 1000))
        

app = Maze()
app.run()
