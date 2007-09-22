// Copyright 2005 The Trustees of Indiana University.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  Authors: Douglas Gregor
//           Andrew Lumsdaine
#ifndef BOOST_PARALLEL_GRAPH_PYTHON_GRAPH_HPP
#define BOOST_PARALLEL_GRAPH_PYTHON_GRAPH_HPP

#include <string>
#include <boost/python.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/python/iterator.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/properties.hpp>

namespace boost { namespace graph { namespace python {

enum graph_doc_kind {
  gd_vertex_iterator,
  gd_num_vertices,
  gd_vertices,
  gd_edge_iterator,
  gd_num_edges,
  gd_edges,
  gd_out_edge_iterator,
  gd_source,
  gd_target,
  gd_out_degree,
  gd_out_edges,
  gd_in_edge_iterator,
  gd_in_degree,
  gd_in_edges,
  gd_adjacency_iterator,
  gd_adjacent_vertices,
  gd_add_edge,
  gd_remove_edge,
  gd_add_vertex,
  gd_clear_vertex,
  gd_remove_vertex,
  gd_last
};

extern const char* graph_docs[gd_last];

using boost::python::class_;

namespace detail {
  
  template<typename Graph>
  inline void 
  maybe_index_vertex(typename graph_traits<Graph>::vertex_descriptor u,
                     Graph& g, mpl::true_)
  {
    put(vertex_index, g, u, num_vertices(g));
  }

  template<typename Graph>
  inline void 
  maybe_index_vertex(typename graph_traits<Graph>::vertex_descriptor,
                     Graph&, mpl::false_)
  {
  }

  template<typename Graph>
  void maybe_reindex_vertices(Graph& g, mpl::true_)
  {
    typename graph_traits<Graph>::vertices_size_type index = 0;
    BGL_FORALL_VERTICES_T (v, g, Graph)
      put(vertex_index, g, v, index++);
  }

  template<typename Graph>
  inline void maybe_reindex_vertices(Graph& g, mpl::false_)
  {
  }

  template<typename Graph>
  inline void 
  maybe_index_edge(typename graph_traits<Graph>::edge_descriptor e,
                     Graph& g, mpl::true_)
  {
    put(edge_index, g, e, num_edges(g));
  }

  template<typename Graph>
  inline void 
  maybe_index_edge(typename graph_traits<Graph>::edge_descriptor,
                   Graph&, mpl::false_)
  {
  }

  template<typename Graph>
  void maybe_reindex_edges(Graph& g, mpl::true_)
  {
    typename graph_traits<Graph>::edges_size_type index = 0;
    BGL_FORALL_EDGES_T (e, g, Graph)
      put(edge_index, g, e, index++);
  }

  template<typename Graph>
  inline void maybe_reindex_edges(Graph& g, mpl::false_)
  {
  }
}

template<typename Graph>
class graph
{
  typedef graph_traits<Graph> Traits;
  typedef typename Traits::vertex_descriptor vertex_descriptor;
  typedef typename Traits::edge_descriptor   edge_descriptor;

  template<typename T>
  static boost::python::object pod_getstate(const T& value)
  {
    using boost::python::list;

    list bytes;
    const char* data = reinterpret_cast<const char*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i)
      bytes.append(data[i]);
    return bytes;
  }

  template<typename T>
  static void pod_setstate(T& value, boost::python::object state)
  {
    using boost::python::list;
    using boost::python::extract;

    list bytes = extract<list>(state);
    char* data = reinterpret_cast<char*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i)
      data[i] = extract<char>(bytes[i]);
  }

 public:
  template<typename T, typename Basis, typename HeldType, typename NonCopyable>
  graph(class_<T, Basis, HeldType, NonCopyable>&)
  {
    using boost::python::class_;
    using boost::python::self;

#ifdef BOOST_MSVC
#define BGL_PYTHON_HACK graph<Graph>::
#else
#define BGL_PYTHON_HACK
#endif
    if (!detail::type_already_registered<vertex_descriptor>())
      class_<vertex_descriptor>("Vertex")
        .def(self == self)
        .def(self != self)
        .enable_pickling()
        .def("__getstate__", &BGL_PYTHON_HACK pod_getstate<vertex_descriptor>)
        .def("__setstate__", &BGL_PYTHON_HACK pod_setstate<vertex_descriptor>)
        ;
    
    if (!detail::type_already_registered<edge_descriptor>())
      class_<edge_descriptor>("Edge")
        .def(self == self)
        .def(self != self)
        .enable_pickling()
        .def("__getstate__", &BGL_PYTHON_HACK pod_getstate<edge_descriptor>)
        .def("__setstate__", &BGL_PYTHON_HACK pod_setstate<edge_descriptor>)
        ;
#undef BGL_PYTHON_HACK
  }
};

template<typename Graph>
class vertex_list_graph
{
  typedef graph_traits<Graph> Traits;
  typedef typename Traits::vertices_size_type vertices_size_type;
  typedef typename Traits::vertex_iterator    vertex_iterator;

  static vertices_size_type py_num_vertices(const Graph& g)
  {
    return num_vertices(g);
  }

  static simple_python_iterator<vertex_iterator> py_vertices(const Graph& g)
  {
    return simple_python_iterator<vertex_iterator>(vertices(g));
  }

public:
  template<typename T, typename Basis, typename HeldType, typename NonCopyable>
  vertex_list_graph(class_<T, Basis, HeldType, NonCopyable>& graph)
  {
    simple_python_iterator<vertex_iterator>
      ::declare("VertexIterator", graph_docs[gd_vertex_iterator]);
    graph.def("num_vertices", &py_num_vertices, graph_docs[gd_num_vertices])
         .add_property("vertices", &py_vertices, graph_docs[gd_vertices]);
  }
};

template<typename Graph>
class edge_list_graph
{
  typedef graph_traits<Graph> Traits;
  typedef typename Traits::edges_size_type edges_size_type;
  typedef typename Traits::edge_iterator   edge_iterator;

  static edges_size_type py_num_edges(const Graph& g)
  {
    return num_edges(g);
  }

  static simple_python_iterator<edge_iterator> py_edges(const Graph& g)
  {
    return simple_python_iterator<edge_iterator>(edges(g));
  }

public:
  template<typename T, typename Basis, typename HeldType, typename NonCopyable>
  edge_list_graph(class_<T, Basis, HeldType, NonCopyable>& graph)
  {
    simple_python_iterator<edge_iterator>
      ::declare("EdgeIterator", graph_docs[gd_edge_iterator]);
    graph.def("num_edges", &py_num_edges, graph_docs[gd_num_edges])
         .add_property("edges", &py_edges, graph_docs[gd_edges]);
  }
};

template<typename Graph>
class incidence_graph
{
  typedef graph_traits<Graph> Traits;
  typedef typename Traits::out_edge_iterator out_edge_iterator;
  typedef typename Traits::degree_size_type degree_size_type;
  typedef typename Traits::vertex_descriptor vertex_descriptor;
  typedef typename Traits::edge_descriptor   edge_descriptor;

  static vertex_descriptor py_source(const Graph& g, edge_descriptor edge)
  {
    return source(edge, g);
  }

  static vertex_descriptor py_target(const Graph& g, edge_descriptor edge)
  {
    return target(edge, g);
  }

  static degree_size_type py_out_degree(const Graph& g, vertex_descriptor u)
  {
    return out_degree(u, g);
  }

  static simple_python_iterator<out_edge_iterator> 
  py_out_edges(const Graph& g, vertex_descriptor u)
  {
    return simple_python_iterator<out_edge_iterator>(out_edges(u, g));
  }

public:
  template<typename T, typename Basis, typename HeldType, typename NonCopyable>
  incidence_graph(class_<T, Basis, HeldType, NonCopyable>& graph)
  {
    simple_python_iterator<out_edge_iterator>
      ::declare("OutEdgeIterator", graph_docs[gd_out_edge_iterator]);
    graph.def("source", &py_source, graph_docs[gd_source])
         .def("target", &py_target, graph_docs[gd_target])
         .def("out_degree", &py_out_degree, graph_docs[gd_out_degree])
         .def("out_edges", &py_out_edges, graph_docs[gd_out_edges]);
  }
};

template<typename Graph>
class bidirectional_graph
{
  typedef graph_traits<Graph> Traits;
  typedef typename Traits::in_edge_iterator in_edge_iterator;
  typedef typename Traits::degree_size_type degree_size_type;
  typedef typename Traits::vertex_descriptor vertex_descriptor;
  typedef typename Traits::edge_descriptor   edge_descriptor;

  static degree_size_type py_in_degree(const Graph& g, vertex_descriptor u)
  {
    return in_degree(u, g);
  }

  static simple_python_iterator<in_edge_iterator> 
  py_in_edges(const Graph& g, vertex_descriptor u)
  {
    return simple_python_iterator<in_edge_iterator>(in_edges(u, g));
  }

public:
  template<typename T, typename Basis, typename HeldType, typename NonCopyable>
  bidirectional_graph(class_<T, Basis, HeldType, NonCopyable>& graph)
  {
    simple_python_iterator<in_edge_iterator>
      ::declare("InEdgeIterator", graph_docs[gd_in_edge_iterator]);
    graph.def("in_degree", &py_in_degree, graph_docs[gd_in_degree])
         .def("in_edges", &py_in_edges, graph_docs[gd_in_edges]);
  }
};

template<typename Graph>
class adjacency_graph
{
  typedef graph_traits<Graph> Traits;
  typedef typename Traits::adjacency_iterator adjacency_iterator;
  typedef typename Traits::vertex_descriptor  vertex_descriptor;

  static simple_python_iterator<adjacency_iterator> 
  py_adjacent_vertices(const Graph& g, vertex_descriptor u)
  {
    return simple_python_iterator<adjacency_iterator>(adjacent_vertices(u, g));
  }

public:
  template<typename T, typename Basis, typename HeldType, typename NonCopyable>
  adjacency_graph(class_<T, Basis, HeldType, NonCopyable>& graph)
  {
    simple_python_iterator<adjacency_iterator>
      ::declare("AdjacencyIterator", graph_docs[gd_adjacency_iterator]);
    graph.def("adjacent_vertices", &py_adjacent_vertices, 
              graph_docs[gd_adjacent_vertices]);
  }
};

template<typename Graph,
         bool AutoIndexVertices = false, 
         bool AutoIndexEdges = false>
class mutable_graph
{
  typedef graph_traits<Graph> Traits;
  typedef typename Traits::vertex_descriptor vertex_descriptor;
  typedef typename Traits::edge_descriptor   edge_descriptor;

  static edge_descriptor
  py_add_edge(Graph& g, vertex_descriptor u, vertex_descriptor v)
  {
    edge_descriptor e = add_edge(u, v, g).first;
    detail::maybe_index_edge(e, g, mpl::bool_<AutoIndexEdges>());
    return e;
  }

  // TBD: remove_edge(u, v)

  static void
  py_remove_edge(Graph& g, edge_descriptor e)
  {
    remove_edge(e, g);
    detail::maybe_reindex_edges(g, mpl::bool_<AutoIndexEdges>());
  }

  // TBD: remove_edge(iter, g)
  // TBD: remove_edge_if(p, g)
  // TBD: remove_out_edge_if(u, p, g)
  // TBD: remove_in_edge_if(u, p, g)

  static vertex_descriptor py_add_vertex(Graph& g)
  {
    vertex_descriptor v = add_vertex(g);
    detail::maybe_index_vertex(v, g, mpl::bool_<AutoIndexVertices>());
    return v;
  }

  static void py_clear_vertex(Graph& g, vertex_descriptor u)
  {
    clear_vertex(u, g);
    detail::maybe_reindex_edges(g, mpl::bool_<AutoIndexEdges>());
  }

  static void py_remove_vertex(Graph& g, vertex_descriptor u)
  {
    remove_vertex(u, g);
    detail::maybe_reindex_vertices(g, mpl::bool_<AutoIndexVertices>());
  }

public:
  template<typename T, typename Basis, typename HeldType, typename NonCopyable>
  mutable_graph(class_<T, Basis, HeldType, NonCopyable>& graph)
  {
    graph.def("add_edge", &py_add_edge, graph_docs[gd_add_edge])
         .def("remove_edge", &py_remove_edge, graph_docs[gd_remove_edge])
         .def("add_vertex", &py_add_vertex, graph_docs[gd_add_vertex])
         .def("clear_vertex", &py_clear_vertex, graph_docs[gd_clear_vertex])
         .def("remove_vertex", &py_remove_vertex, graph_docs[gd_remove_vertex])
      ;
  } 
};

} } } // end namespace boost::graph::python

#endif // BOOST_PARALLEL_GRAPH_PYTHON_GRAPH_HPP
