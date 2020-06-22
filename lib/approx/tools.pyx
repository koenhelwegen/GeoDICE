cimport numpy as np
import numpy as np
cimport tools


cpdef np.ndarray project_to_domain(np.ndarray x, np.ndarray domain_low,
                                   np.ndarray domain_high):
    return np.max([domain_low, np.min([domain_high, x], axis=0)], axis=0)
