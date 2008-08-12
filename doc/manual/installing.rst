.. highlight:: sh

Installing Hedge
================

This tutorial will walk you through the process of building
:mod:`hedge`. To follow, you really only need three basic things:

* A UNIX-like machine with web access.
* A C++ compiler, preferably a Version 4.x gcc.
* A working `Python <http://www.python.org>`_ installation, 
  Version 2.4 or newer.

Hedge *can* take advantage of some extra packages:

* `BLAS <http://netlib.org/blas>`_ (will result in substantial speedup)
* `libsilo <https://wci.llnl.gov/codes/silo/>`_ (write Silo visualization files)
* `MPI <http://www.mpi-forum.org>`_ (compute in parallel), 
  via Boost.MPI and its Python wrapper.
* `CUDA <http://nvidia.com/cuda>`_ 
  (compute on graphics cards).
  via `PyCuda <http://mathema.tician.de/software/pycuda>`_.

In this tutorial, we will build a basic version of hedge that does not
need any of these.

Step 1: Build Boost
-------------------

You may already have a working copy of the `Boost C++ libraries
<http://www.boost.org>`_. If so, make sure that it's version 1.35.0 or
newer. If not, no problem, we'll build it now. Before you start, make
sure you have the Python headers (i.e. development information)
installed. Your operating system may call this package something like
`python-dev` or `python-devel`. Next, `download
<http://boost.org/users/download/>`_ the boost release tar.bz2 file.
Then, do this::

    $ tar xfj ~/download/boost_1_35_0.tar.bz2
    $ cd boost_1_35_0
    $ ./configure --prefix=$HOME/pool
    $ make
    $ make install

(Whenever you see the "`$`" dollar sign, this means you should enter
the subsequent text at your shell prompt. You don't have to be `root`.
A few spots are marked with "sudo" to show that these *do* require
root privileges *if* you are using a Python interpreter that is
installed globally.)

You may adapt the file and directory names to suit your liking,
however the rest of this tutorial will assume that you use these
paths.

.. note:: 
  
    Please make sure that the Boost.Python configuration process finds
    the version of Python you intend to use. It is output during the
    configure/make stage.

If you see something like::

    ...failed updating 30 targets...
    ...skipped 2 targets...

at the end of the build process, please double-check that you have the
Python headers installed. If you failed fewer targets (up to 5),
you're probably ok for hedge, but you might still want to install
`libz-dev` and `libbz2-dev` for that "perfect score".

Step 1.5: Tell the Dynamic Linker about Boost
---------------------------------------------

If you use a bash or /bin/sh or another POSIX-compliant shell, use
this command::

    export LD_LIBRARY_PATH=$HOME/pool/lib:${LD_LIBRARY_PATH}

or, if you are still using a C Shell, use this::

    setenv LD_LIBRARY_PATH $HOME/pool/lib:${LD_LIBRARY_PATH}

You might want to put this command in your startup script, so you
don't have to type this over and over. If you forget this step, you
will see errors like this one later on::

    ...gibberish...
    ImportError: libboost_python-gcc42-mt-1_35.so.1.35.0: 
    cannot open shared object file: No such file or directory
    ...gibberish...

Step 2: Install Boost.Bindings
------------------------------

Download the most recent release of the Boost Bindings from `here
<http://mathema.tician.de/software/boost-numeric-bindings>`_ and
type::

    $ tar xfz ~/download/boost-bindings-20YYMMDD.tar.gz
    $ cd boost-bindings
    $ ./configure --prefix=$HOME/pool
    $ make install

Change the "YYMMDD" to match the release you downloaded. Note that
this is a header-only library, so all it needs to do for installation
is copy some files.

Step 3: Create and Customize a Configuration File
-------------------------------------------------

Copy and paste the following text into a file called
:file:`.aksetup-defaults.py` (Make sure not to miss
the initial dot, it's important.) in your home directory::

    BOOST_BINDINGS_INC_DIR = ['/home/andreas/pool/include/boost-bindings']
    BOOST_INC_DIR = ['/home/andreas/pool/include/boost-1_35']
    BOOST_LIB_DIR = ['/home/andreas/pool/lib']
    BOOST_PYTHON_LIBNAME = ['boost_python-gcc42-mt']

You will need to adapt the path names in this file to your personal
situation, of course.

Additionally, make sure that the compiler tag in
`BOOST_PYTHON_LIBNAME` matches your boost libraries. (It's `gcc42` in
the example, which stands for gcc Version 4.2. Yours may be different.
Find out by looking at the directory listing of :file:`$HOME/pool/lib`, or
wherever you installed the Boost libraries.)

Optional: Tell :mod:`hedge` about BLAS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have the `Basic Linear Algebra Subroutines <http://netlib.org/blas>`_
or a tuned implementation such as `ATLAS <http://math-atlas.sf.net>`_ available,
you may tell hedge about their presence for a big speed boost. Simply add the 
following three lines to the file :file:`.aksetup-defaults.py` in your home
directory that you created just now::

    HAVE_BLAS = True
    BLAS_LIB_DIR = ['/where/ever/your/blas/is']
    BLAS_LIBNAME = ['your_blas_libname'] # without leading lib and trailing .a/.so

If you are using ATLAS, you may need to specify a combination of
libraries similar to these::

    BLAS_LIBNAME = ['f77blas', 'atlas', 'gfortran'] # example if using atlas

Step 4: Download and Unpack hedge
---------------------------------

Download the latest `release of hedge
<http://pypi.python.org/pypi/hedge>`_. Then do this::

    $ tar xfz hedge-VERSION.tar.gz

Step 5: Install Numpy
---------------------

If you don’t already have `numpy <http://numpy.org>`_ installed, this
is an easy way to install it::

    $ cd hedge-VERSION
    $ sudo python ez_setup.py # this will install setuptools
    $ sudo easy_install numpy # this will install numpy using setuptools

Note that installing numpy can take a few minutes, this is normal.

Step 6: Build and Install hedge
-------------------------------

Actually compiling and installing hedge should now be fairly simple::

    $ cd hedge-VERSION # if you're not there already
    $ sudo python setup.py install

Get some coffee while hedge and its dependencies are installed. If
you get no errors, congratulations! You have successfully built hedge.

For your information: This step in the installation will automatically
download and install (or update) the correct versions of the following
packages:

 * Pytools
 * Pymbolic
 * PyUblas
 * MeshPy

Success! So what now?
---------------------

One of the first things you might want to try is running hedge's unit tests. Follow me::

    $ cd hedge-VERSION/test
    $ python test_hedge.py
    ...............................
    ----------------------------------------------------------------------
    Ran 31 tests in 35.187s

    OK

Once that succeeds, you're all set. Next, I'd suggest you go and play
with the wave equation example for a bit::

    $ cd hedge-VERSION/examples/wave
    $ python wave-min.py
    (stuff happens for a little while)

Now download `VisIt <https://wci.llnl.gov/codes/visit/>`_ and marvel
at the output. :) Then, point your editor at `wave-min.py` and start
tinkering.

Have fun!
