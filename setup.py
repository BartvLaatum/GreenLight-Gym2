import os

from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext as _build_ext
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define the path for the C++ module
module_path = "gl_gym/environments/models/greenlight_model.cpp"

ext_modules = [
    Pybind11Extension(
        "gl_gym.environments.models.greenlight_model",
        [module_path],
        include_dirs=[
            ".",                                    # Add current directory
            "/usr/local/include",                   
            "/home/bart/anaconda3/envs/gl_gym/include/python3.11",
            "/home/bart/anaconda3/envs/gl_gym/lib/python3.11/site-packages/pybind11/include",
        ],
        libraries=["casadi"],
        library_dirs=[
            "/usr/local/lib",                       # CasADi library directory
        ],
        extra_compile_args=[
            "-std=c++17",        # C++ standard flag
            "-O3"                # Optimization flag for speed
        ],        extra_link_args=[
            "-L/usr/local/lib",                     # Linker flag to find libcasadi.so
            "-Wl,-rpath,/usr/local/lib"             # Runtime library path
        ],
    )
]

# Custom build_ext class to change the output directory
class build_ext(_build_ext):
    def build_extensions(self):
        # Ensure the output directory exists
        os.makedirs(self.build_lib, exist_ok=True)
        # Call the original build_extensions method
        super().build_extensions()

# Custom command to build only the Cython extensions
class BuildCPPOnlyCommand(Command):
    description = "Build only the C++ extensions."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Build the Cython extensions
        self.run_command('build_ext')

# Function to read the basic requirements file
def read_requirements():
    with open('requirements.txt') as req_file:
        return req_file.read().splitlines()


setup(
    name="gl_gym",
    version="0.1",
    description="Gymnasium environment for greenhouse climate control, underlying dynamical model in C++ with Python bindings.",
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': build_ext,          # Use the custom build_ext class
    },
    
    install_requires=read_requirements(),  # Basic dependencies
)
