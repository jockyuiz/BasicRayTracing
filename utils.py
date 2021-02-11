import numpy as np
from PIL import Image


def vec(list):
    """Handy shorthand to make a single-precision float array."""
    return np.array(list, dtype=np.float32)

def normalize(v):
    """Return a unit vector in the direction of the vector v."""
    return v / np.linalg.norm(v)


def from_srgb(img_srgb):
    return np.where(img_srgb > 0.04045, ((img_srgb + 0.055) / 1.055)**2.4, img_srgb / 12.92).astype(np.float32)

def to_srgb(img):
    img_clip = np.clip(img, 0, 1)
    return np.where(img > 0.0031308, (1.055 * img_clip**(1/2.4) - 0.055), 12.92 * img_clip)

def from_srgb8(img_srgb8):
    return from_srgb(img_srgb8 / 255.0)

def to_srgb8(img):
    return np.clip(np.round(255.0 * to_srgb(img)), 0, 255).astype(np.uint8)


def save_image(img, fname, wp):
    img_ui8 = to_srgb8(img / wp)
    Image.fromarray(img_ui8[::-1,:,:], 'RGB').save(fname)

def load_image(fname):
    img_ui8 = np.array(Image.open(fname))[::-1,:,:3]
    return from_srgb8(img_ui8)



def read_obj(f):
    """Read a file in the Wavefront OBJ file format.

    Argument is an open file.
    Returns a tuple of NumPy arrays: (indices, positions, normals, uvs).
    """
    
    # position, normal, uv, and face data in the order they appear in the file
    f_posns = []
    f_normals = []
    f_uvs = []
    f_faces = []

    # set of unique index combinations that appear in the file
    verts = set()

    # read file
    for words in (line.split() for line in f.readlines()):
        if words[0] == 'v':
            f_posns.append([float(s) for s in words[1:]])
        elif words[0] == 'vn':
            f_normals.append([float(s) for s in words[1:]])
        elif words[0] == 'vt':
            f_uvs.append([float(s) for s in words[1:]])
        elif words[0] == 'f':
            if len(words) > 4:
                print('read_obj error: non-triangular face encountered.')
                return (np.zeros((0,3), np.int32), np.zeros((0,3), np.float32), None, None)
            f_faces.append(words[1:])
            for w in words[1:]:
                verts.add(w)
    f.close()

    # there is one vertex for each unique index combo; number them
    vertmap = dict((s,i) for (i,s) in enumerate(sorted(verts)))

    # collate the vertex data for each vertex
    posns = [None] * len(vertmap)
    normals = [None] * len(vertmap)
    uvs = [None] * len(vertmap)
    for k, v in vertmap.items():
        w = k.split('/')
        posns[v] = f_posns[int(w[0]) - 1]
        if len(w) > 1 and w[1]:
            uvs[v] = f_uvs[int(w[1]) - 1]
        if len(w) > 2 and w[2]:
            normals[v] = f_normals[int(w[2]) - 1]

    # set up faces using our ordering
    inds = [[vertmap[k] for k in f] for f in f_faces]

    # convert all to NumPy arrays with the right datatypes
    return (
        np.array(inds, dtype=np.int32), 
        np.array(posns, dtype=np.float32), 
        np.array(normals, dtype=np.float32) if normals.count(None) == 0 else None,
        np.array(uvs, dtype=np.float32) if uvs.count(None) == 0 else None
        )


def read_obj_triangles(f):
    """Read a file in the Wavefront OBJ file format and convert to separate triangles.

    Argument is an open file.
    Returns an array of shape (n, 3, 3) that has the 3D vertex positions of n triangles.
    """

    (i, p, n, t) = read_obj(f)
    return p[i,:]

def batch_intersect(vs, ray, eps=1e-10):
    """Compute intersections between one ray and a batch of triangles.

    Parameters:
        vs (n, 3, 3) -- the vertices of the n triangles: [k,i,j] is the jth coordinate
            of the ith vertex of the kth triangle
        ray : Ray -- the ray to be intersected
        eps : double -- tolerance for near misses
    Returns:
        t, beta, gamma, i : double, int -- t value of first intersection and which triangle was hit

    Misses are reported with t = infinity, beta = gamma = 0, and i = -1
    """

    # Form array of matrix equations
    A = np.zeros((vs.shape[0], 3, 3), np.float64)
    A[:,:,0] = vs[:,0] - vs[:,1]
    A[:,:,1] = vs[:,0] - vs[:,2]
    A[:,:,2] = [ray.direction]
    b = vs[:,0] - ray.origin

    # Remove any singular equations
    non_sing = np.abs(np.linalg.det(A)) > eps
    A = A[non_sing]
    b = b[non_sing]

    betas, gammas, ts = np.linalg.solve(A, b).transpose()

    # Filter for hits that are inside triangles and in the ray's t range
    hits = np.logical_and(
        np.logical_and(np.logical_and(betas >= -eps, gammas >= -eps), betas + gammas < 1+eps),
        np.logical_and(ts >= ray.start, ts <= ray.end)
    )
    t_hit = ts[hits]

    if len(t_hit) > 0:
        i_hit = np.argmin(t_hit)
        # Remember to map the index back to the original array
        i = np.arange(vs.shape[0])[non_sing][hits][i_hit]
        t = t_hit[i_hit]
        beta = betas[hits][i_hit]
        gamma = gammas[hits][i_hit]
        return t, beta, gamma, i
    else:
        return np.inf, 0, 0, -1
