vars = [
    ("BOOST_INC_DIR", None,
        "The include directory for all of Boost C++"),
    ("BOOST_LIB_DIR", None,
        "The library directory for all of Boost C++"),
    ("BOOST_PYTHON_LIBNAME", "boost_python-gcc41-mt",
        "The name of the Boost Python library binary (without lib and .so)"),
    ("BOOST_BINDINGS_INC_DIR", None,
        "The include directory for the Boost bindings library"),
    ("BOOST_MATH_TOOLKIT_INC_DIR", None,
        "The include directory for the Boost math toolkit"),

    ("USE_SILO", True,
        "Whether to use libsilo for output"),
    ("SILO_INC_DIR", None,
        "The include directory for libsilo"),
    ("SILO_LIB_DIR", None,
        "The library directory for libsilo"),

    ("CXXFLAGS", None,
        "Any extra C++ compiler options to include"),
    ]

subst_files = ["Makefile", "siteconf.py"]
