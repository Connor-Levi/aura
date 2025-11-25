import cupy as cp
from PIL import Image
from scene import scene
from camera import camera
from objects import sphere, light
from util import blackhole


# image dimensions
height = 1080
width = 1920

# scene parameters
epsilon = 0.5
steps = 5

bh = True   # blackhole present or not
distortion = 150    # strength of distortion

# black hole positions (relative to the image dimensions)
bx = width / 2
by = height / 2

# specifying camera
eye = camera(pos = [0, 0, 0], 
        point = [1, 0, 0], 
        up = [0, 1, 0],
        fov = 100,
        width = width, 
        height = height)

# defining the objects to add in the scene
ball = sphere(centre = [10, 0, 0],
        radius = 3, 
        color = [202, 156, 230])

ball1 = sphere(centre = [5, 0, 2],
        radius = 0.7, 
        color = [230, 69, 83])

photon = light(direction = [0, 1, 1])

# defining the scene
scene = scene(eye, [ball, ball1], [photon])

# rendering the image
pic = scene.render(steps, epsilon, width, height)

# post-processing (based on whether bh is 0 or 1)
if bh == True:
    # creates an image distorted by the black hole
    distorted = blackhole(pic, bx, by, distortion)
    img = Image.fromarray(cp.asnumpy(distorted))

else:
    # renders the image undistorted
    img = Image.fromarray(cp.asnumpy(pic))

img.show()
