from utils import *
from ray import *
from cli import render

maptex = Material(load_image('textures/map.png'))
gray = Material(vec([0.2, 0.2, 0.2]))

# Read the triangle mesh for a triangulated unit sphere
(i, p, n, t) = read_obj(open("models/uvsphere_smooth.obj"))

scene = Scene([
    Sphere(vec([-0.7,0,0]), 0.5, maptex),
    Mesh(i, 0.5*p + [[0.7, 0.0, 0.0]], n, t, maptex),
    Sphere(vec([0,-40,0]), 39.5, gray),
])

lights = [
    PointLight(vec([12,10,8]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([2,1.7,5]), target=vec([0,0,0]), vfov=17, aspect=16/9)

render(camera, scene, lights)