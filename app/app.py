# type: ignore
from procedural_maze import build_maze
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import numpy as np
from direct.task.Task import Task
from direct.actor.Actor import Actor
import json
import cv2

class Maze(ShowBase):
    
    def setupCameraControls(self, props: WindowProperties):
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
    def loadTexture(self, path: str):
        texture = loader.loadTexture(path)
        texture.setMagfilter(SamplerState.FT_nearest)
        texture.setMinfilter(SamplerState.FT_nearest)

        return texture

    def addText(self, text_str: str, coords: tuple, color: tuple):
        text = TextNode('node name')
        text.setText(text_str)
        text.setTextColor(color)
        textNodePath = aspect2d.attachNewNode(text)
        textNodePath.setScale(0.07)
        textNodePath.setPos(coords)
        cmr12 = loader.loadFont('cmr12.egg')
        text.setFont(cmr12)
    
    def createMaze(self):
        maze_obj = build_maze(self.maze, self.maze_gray, (512, 512))
        
        self.maze_node = render.attachNewNode(maze_obj)
        self.maze_node.setTwoSided(True)
        self.maze_node.setScale(5)
        
    def idx2world(self, idx):
        return ((idx % self.maze.shape[1] + 0.5) * self.maze_node.getSx(), (idx // self.maze.shape[1] + 0.5) * self.maze_node.getSy(), 0)
        
    async def follow_path(self, path, agent):
        self.amangi[agent].loop("run")
        
        maze_length = len(self.maze)
        video_maze_length = self.video.video_width
        
        for node in path:
            target_pos = self.idx2world(node)
            self.amangi[agent].setPos(target_pos)
            self.amangi[agent].lookAt(target_pos)
            await Task.pause(self.maze_gray[node // self.maze.shape[1], node % self.maze.shape[1]] / 256)
        self.amangi[agent].stop()
        self.runner_finished[agent] = True
        return Task.done
    
    async def recurseSetup(self, rotation):
        # Swap video for image of solved maze
        self.video.stop()
        self.maze_node.setTexture(self.rotation_frames[0])
        
        while not all(self.runner_finished.values()):
            await Task.pause(0.5)
        
        self.setupRunners((rotation + 1) % 4)
        return Task.done
        
    def setupRunners(self, rotation):
        # Display the video
        self.video.setTime(max((self.run_data[agent]['finish_timestamps'][rotation-1] for agent in self.run_data)))
        self.maze_node.setTexture(self.video)
        self.video.play()

        maze_length = len(self.maze)
        video_maze_length = self.video.video_width
        
        for agent in self.run_data:
            node = self.run_data[agent]["paths"][rotation][0]
            self.runner_finished[agent] = False
            self.amangi[agent].setPos(self.idx2world(node))
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
        
        self.maze = cv2.imread("generated/maze.png")
        self.maze_gray = cv2.imread("generated/maze_gray.png", cv2.IMREAD_GRAYSCALE)
        self.video = self.loadTexture("generated/maze.mpg")  # imports video onto map
        self.video.stop()
        
        
        self.rotation_frames = []
        for rotation in range(4):
            self.rotation_frames.append(self.loadTexture(f"generated/solved_maze_{rotation}.png"))
        
        self.createMaze()
        self.maze_node.setTexture(self.video)
        
        
        self.run_data = json.load(open("generated/output.json"))
        self.amangi = {}
        self.runner_finished = {}
        y = 0.85
        dy = 0.1
        for agent in self.run_data:
            self.addText(f"{agent}: {round((sum(self.run_data[agent]['times']) * 1000), 2)} ms.", (-1.7, 0, y), tuple(self.run_data[agent]["color"][::-1] + [1]))
            y -= dy
            new_amangus = Actor("models/amangus.glb")
            new_amangus.reparentTo(render)  # reparents amangus to render
            new_amangus.setScale(2)  # scales amangus
            self.amangi[agent] = new_amangus

        self.setupRunners(0)

        self.camera.setPos((self.maze.shape[1] * self.maze_node.getSx() / 2, self.maze.shape[0] * self.maze_node.getSy() / 2, 500))
        self.camera.setHpr((0, -90, 0))
        

app = Maze()
app.run()
