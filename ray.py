import numpy as np
import math

from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : float, (3,) or (h,w,3) -- the diffuse coefficient
          k_s : float, (3,) or (h,w,3) -- the specular coefficient
          p : float or (h,w) -- the specular exponent
          k_m : float, (3,) or (h,w,3) -- the mirror reflection coefficient
          k_a : float, (3,) or (h,w,3) -- the ambient coefficient (defaults to match diffuse color)
        """
        # TODO A5 (Step2) implement this function
        # Check if each property is an array of shape (h, w, 3)
        # If so, then apply the property using the uv coordinates supplied by the geometry.

        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d

    def lookup_wrap(t,uv):
        """Look up the 2d or 3d vector

        Parameters:
          uv : (2,) -- the texture coordinates at the intersection point
          t: k_d or k_s or p or k_m or k_a ():float,(3,),(h,w),or(h,w,3)
        """
        if np.isscalar(t):
            return t

        if len(t.shape) == 2 or len(t.shape) == 3:
            h, w = t.shape[:2]
            u = uv[0]
            v = uv[1]

            i = round(u * w - 0.5).astype(np.int)
            j = round(v * h - 0.5).astype(np.int)
            if len(t.shape) == 2:
                return t[max(0,min(j,w-1)),max(0,min(i,h-1))]
            elif len(t.shape) == 3:
                return t[max(0,min(j,w-1)),max(0,min(i,h-1)),:]
        else:
            return t

class Hit:

    def __init__(self, t, point=None, normal=None, uv=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          uv : (2,) -- the texture coordinates at the intersection point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.uv = uv
        self.material = material

# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # TODO A5 (Step1) implement this function
        # Copy your implementation from A4
        # Then calculate uv coordinates, to be passed into the Hit initializer
        e, d, c, R = ray.origin, ray.direction, self.center, self.radius
        D = (d @ (e - c))**2 - (d @ d) * ((e - c) @ (e - c) - R**2)
        if D < 0:
            return no_hit
        t = (-d @ (e - c) - np.sqrt(D)) / (d @ d)
        if t < ray.start:
            t = (-d @ (e - c) + np.sqrt(D)) / (d @ d)
        if ray.start < t < ray.end:
            point = ray.origin + t * ray.direction
            normal = normalize(point - self.center)
            x,y,z = normal[:]
            u = (np.arctan2(x,z) + np.pi) / (2*np.pi)
            v = 0.5 + (np.arcsin(y)/np.pi)
            uv = vec([u,v])
            hit = Hit(t,point,normal,uv,self.material)
            return hit
        else:
            return no_hit


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material
        self.norm_vec = normalize(np.cross(vs[1] - vs[0], vs[2] - vs[0]))

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # TODO A5 (Step1) implement this function
        # Copy your implementation from A4
        # Then calculate uv coordinates, to be passed into the Hit initializer
        vs = self.vs
        A = np.array([vs[0] - vs[1], vs[0] - vs[2], ray.direction]).transpose()
        b = vs[0] - ray.origin
        beta, gamma, t = np.linalg.solve(A, b)
        if beta >= 0 and gamma >= 0 and beta + gamma < 1 and ray.start < t < ray.end:
            point = ray.origin + t * ray.direction
            uv = vec([beta,gamma])
            return Hit(t, ray.origin + t * ray.direction, self.norm_vec,uv, self.material)
        else:
            return no_hit

class Mesh:

    def __init__(self, inds, posns, normals, uvs, material):
        self.inds = np.array(inds, np.int32)
        self.posns = np.array(posns, np.float32)
        self.normals = np.array(normals, np.float32) if normals is not None else None
        self.uvs = np.array(uvs, np.float32) if uvs is not None else None
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this mesh, if it exists.
           Use the batch_intersect function in the utils package

        Parameters:
          ray : Ray -- the ray to intersect with the mesh
        Return:
          Hit -- the hit data
        """
        # TODO A5 (Step3 and Step4) implement this function
        # For step 4, check if uvs and normals are not None (respectively)
        # If so, then interpolate them
        (i, p, n, tex) = (self.inds,self.posns,self.normals,self.uvs)
        vs = p[i,:]
        (t, beta, gamma, index) = batch_intersect(vs, ray)

        tri = Triangle(vs[index,:],self.material)
        hit = tri.intersect(ray)
        alpha = 1 - beta - gamma
        if self.normals is None and self.uvs is None:
            return hit
        if self.normals is not None:
            vn = n[i,:]
            normal = alpha * vn[index,0,:] + beta * vn[index,1,:] + gamma * vn[index,2,:]
            normal = normalize(normal)
            hit.normal = normal
        if self.uvs is not None:
            vt = tex[i,:]
            uv = alpha * vt[index,0,:] + beta * vt[index,1,:] + gamma * vt[index,2,:]
            hit.uv = uv

        return hit


class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]),
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        # TODO A5 copy implementation from A4
        self.eye = eye
        self.f = 1 / np.tan(vfov/2 * np.pi/180)
        self.aspect = aspect
        w = normalize(eye - target)
        u = normalize(np.cross(up, w))
        v = np.cross(w, u)
        self.M = np.block([[u, 0], [v, 0], [w, 0], [eye, 1]]).transpose()

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the lower left
                      corner of the image and (1,1) is the upper right
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A5 copy your implementation from A4
        dir = np.array([self.aspect * (2*img_point[0] - 1), 2*img_point[1] - 1, -self.f, 0])
        return Ray(self.eye, (self.M @ dir)[:3])


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A5 copy implementation from A4 and modify
        # material parameters need to be looked up by the uv's at the intersection point
        offset = self.position - hit.point
        shad_hit = scene.intersect(Ray(hit.point, offset, 1e-6, 1.))
        if shad_hit.t > 1.0:
            mat = hit.material
            r = np.linalg.norm(offset)
            L = offset / r
            N = normalize(hit.normal)
            irrad = self.intensity * np.maximum(0, N @ L) / (r*r)
            V = -normalize(ray.direction)
            H = normalize(V + L)
            return irrad * (Material.lookup_wrap(mat.k_d,hit.uv) + Material.lookup_wrap(mat.k_s,hit.uv) * np.maximum(0, H @ N) ** Material.lookup_wrap(mat.p,hit.uv))
        else:
            return vec([0,0,0])


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A5 copy implementation from A4 and modify
        # k_a needs to be looked up by the uv's at the intersection point

        La = Material.lookup_wrap(hit.material.k_a,hit.uv) * self.intensity
        return La


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle,  Mesh] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A5 copy your implementation from A4
        hits = (surf.intersect(ray) for surf in self.surfs)
        return min((hit for hit in hits), key=lambda h: h.t, default=no_hit)

def reflect(v, n):
    return 2 * (v @ n) * n - v

MAX_DEPTH = 6


def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # TODO A5 copy implementation from A4 and modify
    # k_m needs to be looked up by the uv's at the intersection point
    direct = sum(light.illuminate(ray, hit, scene) for light in lights)
    reflected = 0
    if Material.lookup_wrap(hit.material.k_m,hit.uv) is not None and depth < MAX_DEPTH:
        refl_dir = reflect(-normalize(ray.direction), hit.normal)
        refl_ray = Ray(hit.point, refl_dir, 5e-5)
        refl_hit = scene.intersect(refl_ray)
        if refl_hit.t < np.inf:
            reflected = Material.lookup_wrap(hit.material.k_m,hit.uv) * shade(refl_ray, refl_hit, scene, lights, depth+1)
        else:
            reflected = Material.lookup_wrap(hit.material.k_m,hit.uv) * scene.bg_color
    return direct + reflected


def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    # TODO A5 copy implementation from A4
    cam_img = np.zeros((ny,nx,3), np.float32)
    for i in range(ny):
        for j in range(nx):
            ray = camera.generate_ray(np.array([(j + 0.5) / nx, (i + 0.5) / ny]))
            hit = scene.intersect(ray)
            cam_img[i,j] = shade(ray, hit, scene, lights,5) if hit.t < np.inf else scene.bg_color
    return cam_img
    #return np.zeros((ny,nx,3), np.float32)
