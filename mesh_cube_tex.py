from utils import *
from ray import *
from cli import render

# For step 3, uncomment line 7 (i.e. use the faces texture) to match the reference image
# This should also give the same textured cube as cube_tex
tan = Material(load_image('textures/pips.png') * [[0.7, 0.7, 0.4]])
# tan = Material(load_image('textures/faces.png'))
gray = Material(vec([0.2, 0.2, 0.2]))

# Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
(i, p, n, t) = read_obj(open("models/test2.obj"))

scene = Scene([
    # Make a big sphere for the floor
    Sphere(vec([0,-40,0]), 39.5, gray),
] + [
    Mesh(i, 0.5*p, None, t, tan),
])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1.7,5]), target=vec([0,0,0]), vfov=25, aspect=16/9)

render(camera, scene, lights)
