# --------------------------------------------------------------------
# Specify your configuration below.
# See documentation for hints.
# --------------------------------------------------------------------

HEDGE_CONF_TEMPLATE_VERSION = 2

# --------------------------------------------------------------------
# Path options
# --------------------------------------------------------------------

BOOST_INCLUDE_DIRS = []
BOOST_LIBRARY_DIRS = [] 
BPL_LIBRARIES = ["boost_python-gcc41-mt"]

BOOST_MATH_TOOLKIT_INCLUDE_DIRS = ["/home/andreas/work/boost-math-toolkit"]

# --------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------
#EXTRA_COMPILE_ARGS = ["-fmessage-length=0", "-Wno-sign-compare"]
EXTRA_COMPILE_ARGS = ["-Wno-sign-compare"]

