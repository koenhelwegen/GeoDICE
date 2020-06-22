from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["./lib/models/lrgd2.pyx"]),
    include_dirs=[numpy.get_include()],
)
setup(
    ext_modules=cythonize("./lib/approx/tools.pyx"), include_dirs=[numpy.get_include()]
)
setup(
    ext_modules=cythonize("./lib/approx/chebyshev.pyx"),
    include_dirs=[numpy.get_include()],
)
setup(
    ext_modules=cythonize("./lib/dp/qfunction.pyx"), include_dirs=[numpy.get_include()]
)
setup(ext_modules=cythonize("./lib/dp/tools.pyx"), include_dirs=[numpy.get_include()])
setup(
    ext_modules=cythonize("./lib/dp/coordinator.pyx"),
    include_dirs=[numpy.get_include()],
)
