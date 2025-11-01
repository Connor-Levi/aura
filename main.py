import cupy as cp
from PIL import Image
from camera import camera
from objects import sphere

def run():

    # parameters
    height = 1080
    width = 1980

    steps = 500
    epsilon = 0.5
    
    # canvas
    pic = cp.zeros((height, width, 3), dtype = cp.uint8)

    # object
    ball = sphere(centre = [10, 0, 0], 
            radius = 3, 
            color = [125, 90, 170])
    
    # camera
    eye = camera(pos = [0, 0, 0],
            point = [1, 0, 0],
            up = [0, 1, 0],
            fov = 100,
            width = width,
            height = height)

    # rays
    rays = eye.rays()
    pixels = cp.broadcast_to(eye.pos, rays.shape).copy()
    
    # arrays
    dist = cp.zeros((eye.height, eye.width), dtype = cp.float32)
    hitbox = cp.zeros((eye.height, eye.width), dtype = cp.bool_)
    
    # iterate using signed distance function
    for i in range(steps):
        d = ball.distance(pixels)
        
        # update hitbox if dist to the object is less than epsilon
        hit = d < epsilon
        hitbox = (hitbox + hit) > 0
        
        # rays which hit have the distance set to 0
        d = cp.where(hit, 0, d)
        # rays move forward if there is some distance left
        pixels = pixels + rays * d[:, :, None]

    pic[hitbox] = ball.color

    img = Image.fromarray(cp.asnumpy(pic))
    img.show()

if __name__ == "__main__":
    run()
        



