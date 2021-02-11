from utils import *
from ray import *
from cli import render

tan = Material(load_image('creative/2.jpg')) #11 perviously
gray = Material(vec([0.2, 0.2, 0.2]))

mirror = Material(vec([0.2, 0.2, 0.2]), k_s=load_image('textures/specular-map.png'),\
        p=load_image('textures/mirror-texture.png'), k_m=1.0)

# Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
#vs_list = 0.5* read_obj_triangles(open("models/cube.obj"))
#cube_scene = 10 * read_obj_triangles(open("models/cube2.obj"))
#cube1 = 5 * read_obj_triangles(open("models/cube.obj"))

# Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
(i, p, n, t) = read_obj(open("models/cube.obj"))

scene = Scene(
    # Make a big sphere for the floor
    [Sphere(vec([0,-40,0]), 39.5, mirror),]+
    [Sphere(vec([-40,0,0]), 39.5, mirror),]+
    [Sphere(vec([0,0,-40]), 39.5, mirror),]+
    [Mesh(i, 0.5*p, None, t, tan),]
)

#scene = Scene(
    # Make a big sphere for the floor
    #[Triangle(vss, mirror) for vss in cube_scene
   # ]
  #  +
    # Make triangle objects from the vertex coordinates
    #[Triangle(vs, tan) for vs in vs_list]
   # [Sphere(vec([0,-40,0]), 39.5, mirror)]+
   # [Sphere(vec([-40,0,0]), 39.5, mirror)]+
   # [Sphere(vec([0,0,-40]), 39.5, mirror)]
#)

lights = [
    PointLight(vec([20,20,20]), vec([500,300,300])), # (24,20,10)  300,250,250
    AmbientLight(0.4),
]
#lights = [
 #   AmbientLight(0.4)
#]

#camera = Camera(vec([3,1.7,5]), target=vec([0,0,0]), vfov=15, aspect=16/9)
camera = Camera(vec([5,5,5]), target=vec([0,0,0]), vfov=20, aspect=4/4) #(x,z,y)

render(camera, scene, lights)