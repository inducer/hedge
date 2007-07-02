# --------------------------------------------------------------------
# Specify your configuration below.
# See documentation for hints.
# --------------------------------------------------------------------

HEDGE_CONF_TEMPLATE_VERSION = 3

# --------------------------------------------------------------------
# Path options
# --------------------------------------------------------------------

BOOST_INCLUDE_DIRS = []
BOOST_LIBRARY_DIRS = [] 
BPL_LIBRARIES = ["boost_python-gcc41-mt"]

BOOST_MATH_TOOLKIT_INCLUDE_DIRS = ["/home/andreas/work/boost-math-toolkit"]

BOOST_BINDINGS_INCLUDE_DIRS = ["/home/andreas/work/boost-sandbox"]

USE_SILO = True

SILO_INCLUDE_DIRS = ["/home/andreas/pool/include"]
SILO_LIBRARY_DIRS = ["/home/andreas/pool/lib"]
SILO_LIBRARIES = ["silo"]

# --------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------
#EXTRA_COMPILE_ARGS = ["-fmessage-length=0", "-Wno-sign-compare"]
EXTRA_COMPILE_ARGS = ["-Wno-sign-compare"]

