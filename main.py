import cupy as cp
from PIL import Image
from scene import scene
from camera import camera
from objects import sphere, light
from util import blackhole

bh = True
distortion = 150

height = 1080
width = 1920
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
        color = [114, 135, 253])

ball1 = sphere(centre = [5, 0, 2],
        radius = 0.7, 
        color = [230, 69, 83])

photon = light(direction = [0, 1, 1])

scene = scene(eye, [ball, ball1], [photon])

pic = scene.render(steps, epsilon, width, height)

bx = width / 2
by = height / 2

if bh == True:
    distorted = blackhole(pic, bx, by, distortion)
    img = Image.fromarray(cp.asnumpy(distorted))

else:
    img = Image.fromarray(cp.asnumpy(pic))

img.show()


