from distutils.core import Extension, setup
from Cython.Build import cythonize


extension = Extension(name="module", sources=["library/module.pyx"])
setup(ext_modules=cythonize(extension, compiler_directives={'language_level' : "3"}))