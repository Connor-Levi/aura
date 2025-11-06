import cupy as cp
from PIL import Image
from scene import scene
from camera import camera
from objects import sphere, light

height = 1080
width = 1980
epsilon = 0.5
steps = 5

eye = camera(pos = [0, 0, 0], 
        point = [1, 0, 0], 
        up = [0, 1, 0],
        fov = 100,
        width = width, 
        height = height)

ball = sphere(centre = [10, 0, 0],
        radius = 3, 
        color = [125, 90, 170])

ball1 = sphere(centre = [5, 0, 2],
        radius = 0.7, 
        color = [120, 55, 55])

photon = light(direction = [0, 1, 1])

scene = scene(eye, [ball, ball1], [photon])

pic = scene.render(steps, epsilon)

img = Image.fromarray(cp.asnumpy(pic))
img.show()

