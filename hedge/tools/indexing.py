"""Indexing helpers."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
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
