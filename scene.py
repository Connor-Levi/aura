import cupy as cp
import numpy as np
from PIL import Image


class scene:
    def __init__(self, eye, objects, lights):
        self.eye = eye
        self.objects = objects
        self.lights = lights

    def render(self, steps, epsilon, width, height):
        bg = Image.open("starrysky.jpg").resize((width, height))
        bg = cp.array(bg)
        pic = bg.copy()
        #pic = cp.full((self.eye.height, self.eye.width, 3), 175, dtype = cp.uint8)

        for obj in self.objects:
            normals, hitbox = obj.hitpixels(self.eye, steps, epsilon)
            
            for light in self.lights:
                light_rays = light.rays()
                
                cosine = cp.sum(normals * light_rays[None, :], axis = -1)
                cosine = cp.clip(cosine, 0, 1) # keeping positive values only

                diff = 250
                clr = (obj.color[None, :] + (diff * cosine[:, None])).astype(cp.float32)
                clr = cp.clip(clr, 0, 255)
                clr = clr.astype(np.uint8)

                pic[hitbox] = clr

        return pic

