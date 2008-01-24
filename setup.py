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

import glob
import os
import os.path
import sys



def main():
    from distutils.core import setup, Extension
    from distutils import sysconfig

    def non_matching_config():
        print "*** The version of your configuration template does not match"
        print "*** the version of the setup script. Please re-run configure."
        sys.exit(1)

    try:
        conf = {}
        execfile("siteconf.py", conf)
    except ImportError:
        print "*** Please run configure first."
        sys.exit(1)

    if "HEDGE_CONF_TEMPLATE_VERSION" not in conf:
        non_matching_config()

    if conf["HEDGE_CONF_TEMPLATE_VERSION"] != 4:
        non_matching_config()

    LIBRARY_DIRS = conf["BOOST_LIBRARY_DIRS"]
    LIBRARIES = conf["BPL_LIBRARIES"]

    EXTRA_DEFINES = {}
    EXTRA_INCLUDE_DIRS = []
    EXTRA_LIBRARY_DIRS = []
    EXTRA_LIBRARIES = []

    if conf["HAVE_MPI"]:
        EXTRA_DEFINES["USE_MPI"] = 1
        EXTRA_DEFINES["OMPI_SKIP_MPICXX"] = 1
        LIBRARIES.extend(conf["BOOST_MPI_LIBRARIES"])

        cvars = sysconfig.get_config_vars()
        cvars["CC"] = conf["MPICC"]
        cvars["CXX"] = conf["MPICXX"]

    INCLUDE_DIRS = [
            "src/bgl-python",
            "src/cpp",
            ] \
            + conf["BOOST_BINDINGS_INCLUDE_DIRS"] \
            + conf["BOOST_INCLUDE_DIRS"]

    conf["BLAS_INCLUDE_DIRS"] = []
    conf["USE_BLAS"] = conf["HAVE_BLAS"]

    def handle_component(comp):
        if conf["USE_"+comp]:
            EXTRA_DEFINES["USE_"+comp] = 1
            EXTRA_INCLUDE_DIRS.extend(conf[comp+"_INCLUDE_DIRS"])
            EXTRA_LIBRARY_DIRS.extend(conf[comp+"_LIBRARY_DIRS"])
            EXTRA_LIBRARIES.extend(conf[comp+"_LIBRARIES"])

    handle_component("BLAS")

    setup(name="hedge",
          version="0.90",
          description="The Hybrid-and-Easy Discontinuous Galerkin Environment",
          author=u"Andreas Kloeckner",
          author_email="inform@tiker.net",
          license = "GPLv3",
          url="http://news.tiker.net/software/hedge",
          packages=["hedge"],
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
                extra_compile_args=conf["EXTRA_COMPILE_ARGS"],
                ),
              ],
          data_files=[("include/hedge", glob.glob("src/cpp/*.hpp"))],
         )




if __name__ == '__main__':
    # hack distutils.sysconfig to eliminate debug flags
    # stolen from mpi4py
    import sys
    if not sys.platform.lower().startswith("win"):
        from distutils import sysconfig

        cvars = sysconfig.get_config_vars()
        cflags = cvars.get('OPT')
        if cflags:
            cflags = cflags.split()
            for bad_prefix in ('-g', '-O', '-Wstrict-prototypes'):
                for i, flag in enumerate(cflags):
                    if flag.startswith(bad_prefix):
                        cflags.pop(i)
                        break
                if flag in cflags:
                    cflags.remove(flag)
            cflags.append("-O3")
            cvars['OPT'] = str.join(' ', cflags)
            cvars["CFLAGS"] = cvars["BASECFLAGS"] + " " + cvars["OPT"]
    # and now call main
    main()
