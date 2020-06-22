import math
cimport numpy as np
import numpy as np
import sys
import traceback

# cpdef double simple_polynomial(int j, double x):
#    return math.cos(j*math.acos(x))


cpdef np.ndarray domain_to_box(np.ndarray x,
                               np.ndarray domain_low,
                               np.ndarray domain_high)

cpdef np.ndarray box_to_domain(np.ndarray x,
                             np.ndarray domain_low,
                             np.ndarray domain_high)

cpdef double simple_polynomial(int j, double x)

cpdef double ndim_polynomial(list J, np.ndarray X)

cpdef double approximation(list J, list coef, np.ndarray x)

cpdef get_index_list(dim, degree, x=*)
