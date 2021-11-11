from sklearn.neighbors import NearestNeighbors
import numpy as np


def match_points(source, target, max_correspondence_search=None):
    if max_correspondence_search is None:
        max_correspondence_search = source.shape[0]
    
    # get the nearest neighbors up to some depth
    # the maximum depth gives the full distance matrix
    # this will guarantee a 1-to-1 correspondence between points by 
    # distance, however it could be very slow for large datasets
    nn = NearestNeighbors(n_neighbors=max_correspondence_search)
    nn.fit(target)
    distances, indicies = nn.kneighbors(source, return_distance=True)
    # this will give us a list of the row and column indicies (in the distance matrix)
    # of the distances in increasing order
    dist_argsort_row, dist_argsort_col = np.unravel_index(distances.ravel().argsort(), distances.shape)
    source_idxs = []
    target_idxs = []
    dists = []
    
    for dar, dac in zip(dist_argsort_row, dist_argsort_col):
        if dar not in source_idxs:
            tidx = indicies[dar, dac]
            if tidx not in target_idxs:
                source_idxs.append(dar)
                target_idxs.append(tidx)
                dists.append(distances[dar, dac])
                
    
    return np.array(dists), np.array(source_idxs), np.array(target_idxs)


def compute_transform(source, target, return_as_single_matrix=False):
    # basic equation we are trying to optimize
    # error = target - scale*Rot*source - translation
    # so we need to find the scale, rotation, and translation 
    # that minimizes the above equation
    
    # based on notes from
    # https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    
    # made more clear from http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    
    # in this implementation I assume that the matricies source and target are of 
    # the form n x m, where n is the number of points in each matrix
    # and m is the number of dimensions
    
    assert source.shape == target.shape
    n, m = source.shape
    
    # compute centroids
    
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    
    # this removes the translation component of the transform
    # we can estimate the translation by the centroids
    source_rel = source - source_centroid
    target_rel = target - target_centroid
    
    source_var = source_rel.var()
    
    # next we need to estimate the covariance matrix between
    # the source and target points (should be an mxm matrix)
    # in the literature this is often denoted as "H"
    
    H = target_rel.T.dot(source_rel) / (n*m)
    
    
    # now we can do SVD on H to get the rotation matrix
    
    U, D, V = np.linalg.svd(H)
    
    # rotation - first check the determinants of U and V
    u_det = np.linalg.det(U)
    v_det = np.linalg.det(V)
    
    S = np.eye(m)
    
    if u_det*v_det < 0.0:
        S[-1] = -1
    
    rot = V.T.dot(S).dot(U)
    # TODO no rotation
    rot = np.eye(3)
    
    # compute the scale
    scale = (np.trace(np.diag(D).dot(S)) / source_var)
    
    # compute the translation
    trans = target_centroid - source_centroid.dot(rot*scale)
    
    if return_as_single_matrix:
        T = np.eye(m+1)
        T[:m,:m] = scale * rot
        T[m,:m] = trans
        return T
    
    
    return rot, trans, scale


def icp(source, target, max_iter=100, tol=1e-6, d_tol=1e-10, max_correspondence_search=None):
    sn, sm = source.shape
    tn, tm = target.shape
    
    assert sm == tm, "Dimensionality of point sets must be equal"
    
    S = source.copy()
    T = target.copy()
    
    # initialize the scale, rot, and translation estimates
    # here we will the respective centroids and scales get an initial correspondence between
    # the two point sets
    # if we just take the raw distances,
    # all the points in the source may map to one target and vice versa
    Sc = ( (S-S.mean(axis=0)) / S.std(axis=0))
    Tc = ( (T-T.mean(axis=0)) / T.std(axis=0))

    d,s_idxs, t_idxs = match_points(Sc, Tc, max_correspondence_search=max_correspondence_search)

    rotation, _, _ = compute_transform( Sc[s_idxs, :], Tc[t_idxs, :] )
    scale = 1.0
    translation = T.mean(axis=0) - S.mean(axis=0).dot(scale * rotation)
    S = S.dot(scale*rotation) + translation

    prev_err = 1e6
    n_pt = 0
    for i in range(max_iter):
    
        # match the closest points based on some distance metric (using the sklearn NearestNeighbor object)
        d,s_idxs,t_idxs = match_points(S, T, max_correspondence_search=max_correspondence_search)

        # estimate the translation/rotation/scaling to match the source to the target
        rotation, translation, scale = compute_transform(S[s_idxs, :], T[t_idxs, :])

        # transform the source (i.e. update the positions of the source)
        S = S.dot(scale*rotation) + translation

        # repeat until convergence or max iterations
        err = np.mean(d)

        if np.abs(prev_err - err) <= tol:
            break
        
        prev_err = err
            
    rotation, translation, scale = compute_transform(source, S) # we know the exact correspondences here
    return rotation, translation, scale
