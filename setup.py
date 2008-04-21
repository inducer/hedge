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

        IncludeDir("NUMPY"),

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
        ])




def main():
    import glob
    from aksetup_helper import hack_distutils, \
            get_config, setup, Extension

    hack_distutils()
    conf = get_config()

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"]

    EXTRA_DEFINES = {}
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

    if conf["NUMPY_INC_DIR"] is None:
        try:
            import numpy
            from os.path import join
            conf["NUMPY_INC_DIR"] = [join(numpy.__path__[0], "core", "include")]
        except:
            pass

    INCLUDE_DIRS = [
            "src/cpp",
            ] \
            + conf["BOOST_BINDINGS_INC_DIR"] \
            + conf["BOOST_INC_DIR"] \
            + conf["NUMPY_INC_DIR"]

    conf["BLAS_INC_DIR"] = []
    conf["USE_BLAS"] = conf["HAVE_BLAS"]

    def handle_component(conf, comp):
        if conf["USE_"+comp]:
            EXTRA_DEFINES["USE_"+comp] = 1
            EXTRA_INCLUDE_DIRS.extend(conf[comp+"_INC_DIR"])
            EXTRA_LIBRARY_DIRS.extend(conf[comp+"_LIB_DIR"])
            EXTRA_LIBRARIES.extend(conf[comp+"_LIBNAME"])

    handle_component(conf, "BLAS")

    setup(name="hedge",
            # metadata
            version="0.90",
            description="The Hybrid-and-Easy Discontinuous Galerkin Environment",
            author=u"Andreas Kloeckner",
            author_email="inform@tiker.net",
            license = "GPLv3",
            url="http://mathema.tician.de/software/hedge",

            # dependencies
            setup_requires=[
                "PyUblas",
                ],
            install_requires=[
                "pytools",
                "PyUblasExt",
                "pymbolic",
                "PyUblasExt",
                ],
            extras_require = {
                "mesh":  ["meshpy"],
                "silo": ["pylo"],
                "parallel": ["pymetis"],
                },

            # build info
            packages=["hedge"],
            zip_safe=False,

            package_dir={"hedge": "src/python"},
            ext_package="hedge",

            ext_modules=[
                Extension("_internal", 
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
                    ),
                ],
            data_files=[("include/hedge", glob.glob("src/cpp/*.hpp"))],
            )




if __name__ == '__main__':
    main()
