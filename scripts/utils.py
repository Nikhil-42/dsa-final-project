import numba # type: ignore
import ffmpeg # type: ignore
import numpy as np


class VideoWriter:
    def __init__(self, filepath, fps, shape, input_args: dict|None = None, output_args: dict|None = None):
        if input_args is None:
            input_args = {}
        if output_args is None:
            output_args = {}
        input_args['framerate'] = fps
        input_args['pix_fmt'] = 'bgr24'
        input_args['s'] = '{}x{}'.format(*shape)
        self.filepath = filepath
        self.shape = shape
        self.input_args = input_args
        self.output_args = output_args
        self.process = (
            ffmpeg
                .input('pipe:', format='rawvideo', **input_args)
                .filter('fps', fps=30, round='up')
                .output(self.filepath, **output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
    
    def write(self, frame): 
        self.process.stdin.write(
            frame.astype(np.uint8).tobytes()
        )

    def release(self):
        self.process.stdin.close()
        self.process.wait()
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.release()


@numba.njit(cache=True, parallel=True)
def build_adj_list(maze: np.ndarray):
    """Build an adjacency list from a maze."""
    
    # Create a mask that filters invalid neighbor positions
    def in_range(x, y):
        return (0 < x) * (x < maze.shape[1]-1) * (0 < y) * (y < maze.shape[0]-1)
    
    # Define an iterator of relative neighbor positions
    neighbor_deltas = np.array(((0, -1), (0, 1), (-1, 0), (1, 0)))
    
    # Conversion factor from pixel height delta. Maps [-256,256] to (0, inf)
    # Divide by 88.7228390619 = ln(floatmax) | 256 / 88.7228390619 = 2.88539
    remap = lambda x: x if x != 255 else np.inf
    
    adj_list = (np.empty((maze.shape[0] * maze.shape[1], 4), dtype=np.float32), np.empty((maze.shape[0] * maze.shape[1], 4), dtype=np.int32))
    
    for i in numba.prange(adj_list[0].shape[0]):
        x, y = i % maze.shape[1], i // maze.shape[1]
        
        # Compute neighbor positions
        neighbors = neighbor_deltas + np.array((x, y))

        adj_list[1][i] = np.array([nx + ny * maze.shape[1] for nx, ny in neighbors])
        adj_list[0][i] = np.array([0.5 * remap(maze[ny, nx]) + 0.5 * remap(maze[y, x]) if in_range(nx, ny) else np.inf for nx, ny in neighbors])
    
    return adj_list

@numba.njit(cache=True)
def idx_to_pos(index, maze_width):
    return index % maze_width, index // maze_width

@numba.njit(cache=True)
def pos_to_idx(pos, maze_width):
    return pos[0] + pos[1] * maze_width 