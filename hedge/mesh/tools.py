"""Mesh helper facilities."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""




# mesh reorderings ------------------------------------------------------------
def cuthill_mckee(graph):
    """Return a Cuthill-McKee ordering for the given graph.

    See (for example)
    Y. Saad, Iterative Methods for Sparse Linear System,
    2nd edition, p. 76.

    *graph* is given as an adjacency mapping, i.e. each node is
    mapped to a list of its neighbors.
    """
    from pytools import argmin

    # this list is called "old_numbers" because it maps a
    # "new number to its "old number"
    old_numbers = []
    visited_nodes = set()
    levelset = []

    all_nodes = set(graph.keys())

    def levelset_cmp(node_a, node_b):
        return cmp(len(graph[node_a]), len(graph[node_b]))

    while len(old_numbers) < len(graph):
        if not levelset:
            unvisited = list(set(graph.keys()) - visited_nodes)

            if not unvisited:
                break

            start_node = unvisited[
                    argmin(len(graph[node]) for node in unvisited)]
            visited_nodes.add(start_node)
            old_numbers.append(start_node)
            levelset = [start_node]

        next_levelset = set()
        levelset.sort(levelset_cmp)

        for node in levelset:
            for neighbor in graph[node]:
                if neighbor in visited_nodes:
                    continue

                visited_nodes.add(neighbor)
                next_levelset.add(neighbor)
                old_numbers.append(neighbor)

        levelset = list(next_levelset)

    return old_numbers
