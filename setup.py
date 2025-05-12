import os
from setuptools import setup, find_packages, Extension
from pybind11 import get_include as get_pybind_include
from pybind11.setup_helpers import build_ext

# Determine CasADi installation prefix (override via env vars if needed)
casadi_prefix = os.environ.get("CASADI_PREFIX", "/usr/local")
casadi_include = os.environ.get("CASADI_INCLUDE", os.path.join(casadi_prefix, "include"))
casadi_lib = os.environ.get("CASADI_LIB", os.path.join(casadi_prefix, "lib"))

# Read runtime requirements from requirements.txt
def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Define C++ extension modules
ext_modules = [
    Extension(
        "gl_gym.environments.models.greenlight_model",
        ["gl_gym/environments/models/greenlight_model.cpp"],
        include_dirs=[get_pybind_include(), casadi_include],
        library_dirs=[casadi_lib],
        libraries=["casadi"],
        language="c++",
        extra_compile_args=["-std=c++17", "-fPIC"],
    ),
    # Add other C++/pybind11 extensions below
]

setup(
    name="gl_gym",
    version="0.1.0",
    description=(
        "Gymnasium environments for greenhouse climate control, "
        "with C++ core via CasADi"
    ),
    packages=find_packages(include=["gl_gym"]),
    install_requires=read_requirements(),
    setup_requires=["pybind11>=2.6.0"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
    zip_safe=False,
)
