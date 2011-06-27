"""Indexing helpers."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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




import numpy
import numpy.linalg as la




def count_subset(subset):
    from pytools import len_iterable
    return len_iterable(uc for uc in subset if uc)




def full_to_subset_indices(subset, base=0):
    """Takes a sequence of bools and turns it into an array of indices
    to be used to extract the subset from the full set.

    Example:

    >>> full_to_subset_indices([False, True, True])
    array([1 2])
    """

    result = []
    for i, is_in in enumerate(subset):
        if is_in:
            result.append(i + base)

    return numpy.array(result, dtype=numpy.intp)



def full_to_all_subset_indices(subsets, base=0):
    """Takes a sequence of bools and generates it into an array of indices
    to be used to extract the subset from the full set.

    Example:

    >>> list(full_to_all_subset_indices([[False, True, True], [True,False,True]]))
    [array([1 2]), array([3 5]
    """

    for subset in subsets:
        result = []
        for i, is_in in enumerate(subset):
            if is_in:
                result.append(i + base)
        base += len(subset)

        yield numpy.array(result, dtype=numpy.intp)



def partial_to_all_subset_indices(subsets, base=0):
    """Takes a sequence of bools and generates it into an array of indices
    to be used to insert the subset into the full set.

    Example:

    >>> list(partial_to_all_subset_indices([[False, True, True], [True,False,True]]))
    [array([0 1]), array([2 3]
    """

    idx = base
    for subset in subsets:
        result = []
        for is_in in subset:
            if is_in:
                result.append(idx)
                idx += 1

        yield numpy.array(result, dtype=numpy.intp)




class IndexListRegistry(object):
    """An index list registry maintains a numbering of index lists.
    (such as for face indices in a volume element)
    There needs to exist a one-to-one mapping of identifiers to 
    index lists. If, upon registration, an index list with a known
    identifier is registered, its index is returned directly.
    Otherwise, a generator thunk is called to generate the list,
    and a new index list number is allocated.
    """

    def __init__(self, debug=False):
        self.index_lists = []
        self.il_id_to_number = {}
        self.il_to_number = {}
        self.debug = debug

    def register(self, identifier, generator):
        try:
            result = self.il_id_to_number[identifier]
            if self.debug:
                assert generator() == self.index_lists[result], (
                        "identifier %s used for two different index lists"
                        % str(identifier))
            return result
        except KeyError:
            il = generator()
            try:
                nbr = self.il_to_number[il]
            except KeyError:
                nbr = len(self.index_lists)
                self.index_lists.append(il)
                self.il_id_to_number[identifier] = nbr
                self.il_to_number[il] = nbr
            else:
                self.il_id_to_number[identifier] = nbr
            return nbr

    def get_list_length(self):
        from pytools import single_valued
        return single_valued(len(il) for il in self.index_lists)





def find_index_map_from_node_sets(old_nodes, new_nodes, threshold=1e-12):
    """Given *old_nodes* and *new_nodes*, which occupy
    the same spots but may have switched identities,
    returns an index tuple, which satisfies (in shorthand)
    `new_nodes[imap] == old_nodes`.
    """
    idx_map = []

    # yay O(n^2)
    for old_node in old_nodes:
        found = False
        for new_idx, new_node in enumerate(new_nodes):
            if la.norm(old_node-new_node) < threshold:
                idx_map.append(new_idx)
                found = True
                break

        if not found:
            raise ValueError("a corresponding node for %s was not found"
                    % old_node)

    return tuple(idx_map)



