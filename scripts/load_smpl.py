import pickle
import numpy as np
import scipy.io
import pdb
import trimesh
pdb.set_trace()

def load_smpl():
    smpl_subdiv_fpath = "./SMPL_subdiv.mat"
    pdist_matrix_fpath = "./SMPL_SUBDIV_TRANSFORM.mat"
    SMPL_subdiv = scipy.io.loadmat(smpl_subdiv_fpath)
    vertices_ = np.array(SMPL_subdiv['vertex']).transpose()
    faces_ = np.array(SMPL_subdiv['faces']).transpose() - 1

    PT = scipy.io.loadmat(pdist_matrix_fpath)
    PT = PT['index'].squeeze() - 1
    verts = np.zeros((27554,3))
    verts[PT,:] = vertices_
    faces = PT[faces_]
    return verts, faces

verts, faces = load_smpl()
trimesh.Trimesh(verts, faces).export('smpl_subdiv.obj')
dict_smpl = {'vertices': verts, 'faces': faces}
trimesh.Trimesh(verts, faces).export('smpl_subdiv.obj')
with open('smpl_27554.pkl', 'wb') as handle:
    pickle.dump(dict_smpl, handle, protocol=pickle.HIGHEST_PROTOCOL)
