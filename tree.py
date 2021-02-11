from utils import *
from ray import *
from cli import render

tree = Material(load_image("textures/trees-normal.png"))
gray = Material(vec([0.2, 0.2, 0.2]))

i, p, n, t = read_obj(open("models/low-poly-tree-1.obj"))
tree1 = Mesh(i, 0.5*p + [[0.3, 0, 0]], n, t, tree)


scene = Scene([
    # Make a big sphere for the floor
    Sphere(vec([0,-60,0]), 60., gray),
] + [
    tree1
])

lights = [
    PointLight(vec([12,10,5]), vec([500,500,500])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1,5]), target=vec([0,1,0]), vfov=30, aspect=16/9)

render(camera, scene, lights)