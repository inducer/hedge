vars = [
    ("BOOST_INC_DIR", None,
        "The include directory for all of Boost C++"),
    ("BOOST_LIB_DIR", None,
        "The library directory for all of Boost C++"),
    ("BOOST_PYTHON_LIBNAME", "boost_python-gcc42-mt",
        "The name of the Boost Python library binary (without lib and .so)"),
    ("BOOST_BINDINGS_INC_DIR", None,
        "The include directory for the Boost bindings library"),
    ("BOOST_MATH_TOOLKIT_INC_DIR", None,
        "The include directory for the Boost math toolkit"),
    # -------------------------------------------------------------------------
    ("HAVE_BLAS", False,
        "Whether to build with support for BLAS"),
    ("BLAS_LIB_DIR", None,
        "Library directory for BLAS"),
    ("BLAS_LIB_NAMES", "blas",
        "Library names for BLAS, comma-separated"),
    # -------------------------------------------------------------------------
    ("HAVE_MPI", False,
        "Whether to build with support for MPI"),
    ("MPICC", "mpicc",
        "Path to MPI C compiler"),
    ("MPICXX", None,
        "Path to MPI C++ compiler (defaults to same as MPICC)"),
    ("BOOST_MPI_LIBNAME", "boost_mpi-gcc42-mt",
        "The name of the Boost MPI library binary (without lib and .so)"),
    # -------------------------------------------------------------------------
    ("CXXFLAGS", None,
        "Any extra C++ compiler options to include"),
    ]

subst_files = ["Makefile", "siteconf.py"]
