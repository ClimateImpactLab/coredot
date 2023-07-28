import numpy as np

def dot_starmap(v1, v2):
    """                                                                                                     
    Uses a starmap (itertools) to apply the mul operator on an izipped (v1,v2)                           
    """
    return sum(starmap(mul, izip(v1, v2)))

def dot_numpy(v1, v2): 
    return np.dot(v1, v2)

def dot_choose(v1, v2):
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        return dot_numpy(v1, v2)
    if isinstance(v1, np.ndarray):
        return dot_numpy(v1, np.array(v2))
    if isinstance(v2, np.ndarray):
        return dot_numpy(np.array(v1), v2)
    return dot_starmap(v1, v2)
