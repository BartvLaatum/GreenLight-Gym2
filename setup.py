import os
import sysconfig
from setuptools import setup, find_packages, Command
from setuptools.command.build_ext import build_ext as _build_ext
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Use dynamic include directories
include_dirs = [
    os.getcwd(),                              # Current directory
    sysconfig.get_path("include"),            # Python's include directory
    # pybind11 include directory (automatically detects platform-specific path)
    # This replaces any hard-coded path for pybind11
]
try:
    # Get pybind11 include path
    import pybind11
    include_dirs.append(pybind11.get_include())
except ImportError:
    pass  # User will get an error if pybind11 is not installed

# Define the path for the C++ module
module_path = "gl_gym/environments/models/greenlight_model.cpp"

ext_modules = [
    Pybind11Extension(
        "gl_gym.environments.models.greenlight_model",
        [module_path],
        include_dirs=include_dirs,
        libraries=["casadi"],
        library_dirs=[
            # Use Python's LIBDIR if available, or let users set $CASADI_LIB if needed
            sysconfig.get_config_var('LIBDIR') or "/usr/local/lib",
            os.environ.get("CASADI_LIB", "/usr/local/lib")
        ],
        extra_compile_args=[
            "-std=c++17",
            "-O3"
        ],
        extra_link_args=[
            "-Wl,-rpath," + (os.environ.get("CASADI_LIB", "/usr/local/lib"))
        ],
    )
]

class build_ext(_build_ext):
    def build_extensions(self):
        os.makedirs(self.build_lib, exist_ok=True)
        super().build_extensions()

class BuildCPPOnlyCommand(Command):
    description = "Build only the C++ extensions."
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        self.run_command('build_ext')

def read_requirements():
    with open('requirements.txt') as req_file:
        return req_file.read().splitlines()

setup(
    name="gl_gym",
    version="0.1",
    description="Gymnasium environment for greenhouse climate control, underlying dynamical model in C++ with Python bindings.",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': build_ext,
    },
    install_requires=read_requirements(),
)
