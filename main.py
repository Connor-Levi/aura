import cupy as cp
from PIL import Image
from camera import camera
from objects import sphere
from objects import light

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
    
    # finding normals where light hit
    normals = ball.normal( pixels[hitbox] )
    
    # introduction parallel beams of light
    photon = light(direction = [0, 1, 1])
    light_rays = photon.rays()

    # calculating the projection of rays on normals
    cosine = cp.sum(normals * light_rays[None, :], axis = -1)

    # for strong contrast
    #clr = (ball.color[None, :] + cosine[:, None]).astype(cp.uint8)

    # using diffuse lighting, but causes weak shadows
    #clr = (ball.color[None, :] + (ball.color[None, :] * cosine[:, None])).astype(cp.uint8)
    
    # custom diffused lighting
    diff = 75 # diffusion factor
    clr = (ball.color[None, :] + (diff * cosine[:, None])).astype(cp.uint8)
    
    # limiting color values between 0 and 255
    clr = cp.clip(clr, 0, 255)
    
    pic[hitbox] = clr
    
    img = Image.fromarray(cp.asnumpy(pic))
    img.show()

if __name__ == "__main__":
    run()
        


