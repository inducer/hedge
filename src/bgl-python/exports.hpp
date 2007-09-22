// Copyright 2005 The Trustees of Indiana University.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  Authors: Douglas Gregor
//           Andrew Lumsdaine
#ifndef BOOST_GRAPH_PYTHON_EXPORTS_HPP
#define BOOST_GRAPH_PYTHON_EXPORTS_HPP

namespace boost { namespace graph { namespace python {

template<typename DirectedS> void export_basic_graph(const char* name);
template<typename Graph> void export_property_maps();

template<typename Graph>
void export_generators(boost::python::class_<Graph>& graph, const char* name);

template<typename Graph>
void export_graphviz(boost::python::class_<Graph>& graph, const char* name);

template<typename Graph>
boost::python::object 
vertex_property_map(Graph& g, const std::string& type);

template<typename Graph>
boost::python::object 
edge_property_map(Graph& g, const std::string& type);

} } } // end namespace boost::graph::python

#endif // BOOST_GRAPH_PYTHON_EXPORTS_HPP
