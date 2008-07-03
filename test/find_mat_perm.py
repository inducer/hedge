import numpy
import numpy.linalg as la




def find_mat_perm(orig, permuted, thresh=1e-13):
    assert orig.shape == permuted.shape
    n = orig.shape[0]
    orig_row_candidate_sets = []
    for new_row in range(n):
        candidates = set()
        for orig_row in range(n):
            if abs(permuted[new_row, new_row]-orig[orig_row,orig_row]) < thresh:
                candidates.add(orig_row)
        orig_row_candidate_sets.append(candidates)

    class CandidateRemoved(Exception):
        pass

    single_candidate_rows = set(
            row for row in range(n)
            if len(orig_row_candidate_sets[row]) == 1)

    def remove_candidate(row, candidate):
        orig_row_candidate_sets[row].remove(candidate)
        if len(orig_row_candidate_sets[row]) == 1:
            single_candidate_rows.add(row)
            fixed_candidate = orig_row_candidate_sets[row]
            for cands in orig_row_candidate_sets:
                cands.discard(fixed_candidate)
        assert orig_row_candidate_sets[row], "removed all candidates"
        #print "remove", row, candidate
        raise CandidateRemoved()

    def test_permutation(p):
        for i in single_candidate_rows:
            if i >= len(p):
                continue
            for j in range(len(p)):
                if abs(orig[p[i],p[j]]-permuted[i,j]) > thresh:
                    remove_candidate(j, p[j])

        for i in set(range(len(p))) - single_candidate_rows:
            for j in range(len(p)):
                if abs(orig[p[i],p[j]]-permuted[i,j]) > thresh:
                    return None

        return p

    def test_permutations(prefix=[]):
        result = test_permutation(prefix)
        if result is None:
            return 
        elif len(prefix) == n: 
            return result

        for possible_orig_row in orig_row_candidate_sets[len(prefix)]:
            if possible_orig_row in prefix:
                continue
            result = test_permutations(prefix+[possible_orig_row])
            if result:
                return result

    while any(len(row_can) > 1 for row_can in orig_row_candidate_sets):
        try:
            p = test_permutations()
            if p:
                #print la.norm(orig[p][:,p]-permuted) 
                assert la.norm(orig[p][:,p]-permuted) < 1e-12
                return p
        except CandidateRemoved:
            pass

    assert False, "should not get here"




def main():
    from hedge.element import TetrahedralElement

    el = TetrahedralElement(3)
    dr, ds, dt = el.differentiation_matrices()
    print find_mat_perm(dr, ds)
    print find_mat_perm(dr, dt)


if __name__ == "__main__":
    main()
