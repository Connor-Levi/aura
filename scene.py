import cupy as cp

class scene:
    def __init__(self, eye, objects, lights):
        self.eye = eye
        self.objects = objects
        self.lights = lights

    def render(self, steps, epsilon):
        pic = cp.full((self.eye.height, self.eye.width, 3), 175, dtype = cp.uint8)

        for obj in self.objects:
            normals, hitbox = obj.hitpixels(self.eye, steps, epsilon)
            
            for light in self.lights:
                light_rays = light.rays()
                
                cosine = cp.sum(normals * light_rays[None, :], axis = -1)
                cosine = cp.clip(cosine, 0, 1) # keeping positive values only

                diff = 200
                clr = (obj.color[None, :] + (diff * cosine[:, None])).astype(cp.uint8)
                clr = cp.clip(clr, 0, 255)

                pic[hitbox] = clr

        return pic

