// Copyright 2005 The Trustees of Indiana University.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  Authors: Douglas Gregor
//           Andrew Lumsdaine
//
// This file is intended to be included multiple times. It lists the
// graph types for which we will generate Python bindings using macros
// that will be defined by the file including this header. The two
// macros are UNDIRECTED_GRAPH(Name,Type) and
// DIRECTED_GRAPH(Name,Type), where Name is the name to expose to
// Python and Type is the C++ type of the graph.
//
// If the DIRECTED_GRAPH macro is undefined, it will be given the same
// definition as UNDIRECTED_GRAPH.

#if !defined(UNDIRECTED_GRAPH)
#  error You must define UNDIRECTED_GRAPH before including this file.
#endif

#if !defined(DIRECTED_GRAPH)
#  define DIRECTED_GRAPH(Name,Type) UNDIRECTED_GRAPH(Name,Type)
#endif

UNDIRECTED_GRAPH(Graph, boost::graph::python::Graph)
DIRECTED_GRAPH(Digraph, boost::graph::python::Digraph)

#undef DIRECTED_GRAPH
#undef UNDIRECTED_GRAPH
