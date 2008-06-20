#!/usr/bin/env python
# -*- coding: latin-1 -*-

# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, \
            Switch, StringListOption

    return ConfigSchema([
        IncludeDir("BOOST", []),
        LibraryDir("BOOST", []),
        Libraries("BOOST_PYTHON", ["boost_python-gcc42-mt"]),

        IncludeDir("BOOST_BINDINGS", []),

        Switch("HAVE_BLAS", False, "Whether to build with support for BLAS"),
        LibraryDir("BLAS", []),
        Libraries("BLAS", ["blas"]),

        Switch("HAVE_MPI", False, "Whether to build with support for MPI"),
        Option("MPICC", "mpicc",
            "Path to MPI C compiler"),
        Option("MPICXX", 
            help="Path to MPI C++ compiler (defaults to same as MPICC)"),
        Libraries("BOOST_MPI", ["boost_mpi-gcc42-mt"]),

        StringListOption("CXXFLAGS", [], 
            help="Any extra C++ compiler options to include"),
        StringListOption("LDFLAGS", [], 
            help="Any extra linker options to include"),
        ])




def main():
    import glob
    from aksetup_helper import hack_distutils, get_config, setup, \
            PyUblasExtension

    hack_distutils()
    conf = get_config(get_config_schema())

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"]

    EXTRA_DEFINES = { "PYUBLAS_HAVE_BOOST_BINDINGS":1 }
    EXTRA_INCLUDE_DIRS = []
    EXTRA_LIBRARY_DIRS = []
    EXTRA_LIBRARIES = []

    if conf["HAVE_MPI"]:
        EXTRA_DEFINES["USE_MPI"] = 1
        EXTRA_DEFINES["OMPI_SKIP_MPICXX"] = 1
        LIBRARIES.extend(conf["BOOST_MPI_LIBNAME"])

        from distutils import sysconfig
        cvars = sysconfig.get_config_vars()
        cvars["CC"] = conf["MPICC"]
        cvars["CXX"] = conf["MPICXX"]

    INCLUDE_DIRS = [
            "src/cpp",
            ] \
            + conf["BOOST_BINDINGS_INC_DIR"] \
            + conf["BOOST_INC_DIR"] \

    conf["BLAS_INC_DIR"] = []
    conf["USE_BLAS"] = conf["HAVE_BLAS"]

    def handle_component(comp):
        if conf["USE_"+comp]:
            EXTRA_DEFINES["USE_"+comp] = 1
            EXTRA_INCLUDE_DIRS.extend(conf[comp+"_INC_DIR"])
            EXTRA_LIBRARY_DIRS.extend(conf[comp+"_LIB_DIR"])
            EXTRA_LIBRARIES.extend(conf[comp+"_LIBNAME"])

    handle_component("BLAS")

    setup(name="hedge",
            # metadata
            version="0.91",
            description="Hybrid Easy Discontinuous Galerkin Environment",
            long_description="""
            hedge is an unstructured, high-order, parallel
            Discontinuous Galerkin solver for partial differential
            equations.   
            
            Features:

            * Supports simplicial unstructured meshes in two and
              three dimensions (i.e. triangles and tetrahedra)
            * Approximates using orthogonal polynomials of any degree
              (and therefore to any order of accuracy) you specify at
              runtime
            * Solves PDEs in parallel using MPI 
            * Easy to use
            * Powerful Parallel Visualization
            """,
            author=u"Andreas Kloeckner",
            author_email="inform@tiker.net",
            license = "GPLv3",
            url="http://mathema.tician.de/software/hedge",
            classifiers=[
              'Environment :: Console',
              'Development Status :: 4 - Beta',
              'Intended Audience :: Developers',
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License (GPL)',
              'Natural Language :: English',
              'Programming Language :: C++',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Physics',
              'Topic :: Scientific/Engineering :: Visualization',
              ],

            # dependencies
            setup_requires=[
                "PyUblas>=0.92.2",
                ],
            install_requires=[
                "PyUblas>=0.92.2",
                "pytools>=3",
                "pymbolic>=0.90",
                "meshpy>=0.91",
                ],
            extras_require = {
                "matrices": ["PyUblasExt>=0.92.7"],
                "silo": ["pylo"],
                "parallel": ["PyMetis>=0.91"],
                },

            # build info
            packages=[
                    "hedge", 
                    "hedge.cuda"
                    ],
            zip_safe=False,

            package_dir={
                    "hedge": "src/python",
                    "hedge.cuda": "src/python/cuda"
                    },
            ext_package="hedge",

            ext_modules=[
                PyUblasExtension("_internal", 
                    ["src/wrapper/wrap_main.cpp", 
                        "src/wrapper/wrap_base.cpp", 
                        "src/wrapper/wrap_special_function.cpp", 
                        "src/wrapper/wrap_flux.cpp", 
                        "src/wrapper/wrap_op_target.cpp", 
                        "src/wrapper/wrap_volume_operators.cpp", 
                        "src/wrapper/wrap_index_map.cpp", 
                        "src/wrapper/wrap_mpi.cpp", 
                        ],
                    include_dirs=INCLUDE_DIRS + EXTRA_INCLUDE_DIRS,
                    library_dirs=LIBRARY_DIRS + EXTRA_LIBRARY_DIRS,
                    libraries=LIBRARIES + EXTRA_LIBRARIES,
                    define_macros=list(EXTRA_DEFINES.iteritems()),
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
                ],
            data_files=[("include/hedge", glob.glob("src/cpp/*.hpp"))],
            )




if __name__ == '__main__':
    main()
