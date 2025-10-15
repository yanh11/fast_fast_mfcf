import heapq
import itertools
from collections import Counter

import numpy as np
import numpy.linalg as LA


# Helper function for debug formatting
def format_frozenset(fs):
    """Convert frozenset to readable format with sorted integer list."""
    if not fs:
        return "[]"
    try:
        # Handle both regular ints and numpy ints
        return str(sorted([int(x) for x in fs]))
    except (ValueError, TypeError):
        return str(sorted(list(fs)))


def format_cliques_list(cliques):
    """Format list of frozenset cliques to readable format."""
    return [format_frozenset(clq) for clq in cliques]


def format_counter(counter):
    """Format Counter with frozenset keys to readable format."""
    if not counter:
        return "{}"
    formatted = {}
    for fs, count in counter.items():
        key = format_frozenset(fs) if fs else "[]"
        formatted[key] = count
    return str(formatted)


def mfcf_control():
    ctl = {
        "min_clique_size": 1,
        "max_clique_size": 4,
        "coordination_number": np.inf,  # maximum allowed uses of a separator
        "cachesize": np.inf,
        "threshold": 0.00,
        "drop_sep": True,
        "method": "MFCF",
    }
    return ctl


class MFCF:
    def __init__(self):
        pass

    def fast_mfcf(
        self, C: np.ndarray, ctl: dict, gf_type: str, cov_matrix: np.ndarray = None
    ):
        self._C = C
        self._ctl = ctl
        self._gf = Gains(C, ctl, gf_type).get_gain

        self._initialise()
        self._compute_mfcf()

        matrix_for_logo = cov_matrix if cov_matrix is not None else C
        J_logo = self._logo(matrix_for_logo, self._cliques, self._separators_count)

        return self._cliques, self._separators_count, self._peo, J_logo

    # Initialisation of data structures

    def _initialise(self):
        self._gains_pq = []
        self._iteration = 0
        self._max_mult = self._ctl.get(
            "max_separator_multiplicity",
            self._ctl.get("coordination_number", float("inf")),
        )

        first_cl = self._get_first_clique()

        self._cliques = [first_cl]
        self._separators_count = Counter()
        self._seen_separators = set()
        self._peo = [v for v in first_cl]  # Perfect elimination order
        self._outstanding_nodes = [
            v for v in range(self._C.shape[0]) if v not in first_cl
        ]

        if self._ctl.get("debug", False):
            self._print_initial_state(first_cl)

        self._process_new_clique_gains(first_cl)

    def _get_first_clique(self) -> frozenset:
        method = self._ctl["method"]
        if method == "TMFG":
            return frozenset(self._first_TMFG())
        elif method == "MFCF":
            return frozenset(self._first_clique())
        else:
            raise ValueError(f"Unknown method: {method}")

    # Start with one node
    def _first_clique(self, first=1):
        C1 = self._C.copy()
        r, c = np.nonzero(self._C <= self._C.mean())
        C1[r, c] = 0
        sums = C1.sum(axis=0)
        cand = np.argsort(-sums, kind="stable")  # stable sort
        clq = cand[:first]
        return clq

    # Same starting clique as TMFG, this guarantees the same final graph as TMFG
    def _first_TMFG(self):
        W = np.square(self._C)
        flat_matrix = W.flatten()
        mean_flat_matrix = np.mean(flat_matrix)
        # Only count weights above the mean
        v = np.sum(W * (W > mean_flat_matrix), axis=1)
        sorted_v = np.argsort(v)[::-1]
        return sorted_v[:4]

    # Main MFCF algorithm loop
    def _compute_mfcf(self):
        while len(self._outstanding_nodes) > 0:
            self._iteration += 1

            gain, v, sep = heapq.heappop(self._gains_pq)
            if self._should_skip_candidate(gain, v, sep):
                continue

            v, sep, parent_clique = self._apply_threshold_and_find_parent(gain, v, sep)
            cliques_before = list(self._cliques)
            new_clique = self._add_new_clique(parent_clique, sep, v)

            self._check_proposed_separator(sep, cliques_before)

            if len(self._outstanding_nodes) == 0:
                break

            self._process_new_clique_gains(new_clique)

    def _should_skip_candidate(self, gain, v, sep):
        if np.isnan(gain) or v not in self._outstanding_nodes:
            return True

        # If drop_sep is enabled, disable candidates with the used separator.
        if self._ctl.get("drop_sep", False):
            if sep in self._seen_separators or self._separators_count[sep] > 0:
                return True

        # length constraint
        minc = self._ctl.get("min_clique_size", 2)
        maxc = self._ctl.get("max_clique_size", 4)
        if not (len(sep) >= minc - 1 and len(sep) < maxc):
            return True

        # multiplicity constraint
        if self._separators_count[sep] >= self._max_mult:
            return True

        # subset-of-some-current-clique constraint
        # TODO: optimize this check
        valid_parent = any(sep.issubset(clq) for clq in self._cliques)
        if not valid_parent:
            return True
        return False

    def _apply_threshold_and_find_parent(self, gain, v, sep):
        pos_gain = -gain  # negate back to positive for threshold compare
        if pos_gain < self._ctl["threshold"]:
            # start a new clique
            v = self._outstanding_nodes[0]
            sep = frozenset()
            parent_clique = frozenset()
        else:
            parent_clique = self._find_parent_clique_for_separator(sep)
        return v, sep, parent_clique

    def _find_parent_clique_for_separator(self, sep):
        for _c in self._cliques:
            if sep <= _c:
                return _c
        return None

    def _add_new_clique(self, parent_clique, sep, v):
        new_clique = frozenset(sep | {v})
        self._peo.append(v)
        self._outstanding_nodes.remove(v)

        if self._ctl.get("debug", False):
            self._print_added_clique(v, new_clique, parent_clique, sep)

        # TODO: optimise
        # --- NEW: keep only maximal cliques (remove any proper subset of the new one) ---
        if len(new_clique) > 1:  # only meaningful if it actually grew
            # Collect proper subset cliques to drop (strict subset)
            to_remove = [c for c in self._cliques if c < new_clique]
            for c in to_remove:
                self._cliques.remove(c)

        # Previous conditional logic simplified: always append the new (possibly maximal) clique
        self._cliques.append(new_clique)
        return new_clique

    # CORRECTED: MFCF separator logic - use proposed sub-clique as separator
    # The separator is simply the proposed sub-clique, unless it equals an
    # entire existing clique
    def _check_proposed_separator(self, proposed_separator, cliques_before):
        if not proposed_separator:
            return  # Empty separator, nothing to record
        self._seen_separators.add(proposed_separator)
        minc = self._ctl.get("min_clique_size", 2)
        maxc = self._ctl.get("max_clique_size", 4)
        sep_len = len(proposed_separator)
        if not (sep_len >= minc - 1 and sep_len < maxc):
            return False
        is_proper_separator = True
        for (
            existing_clique
        ) in cliques_before:  # Check against cliques before adding new one
            if proposed_separator >= existing_clique:  # Not a proper subset
                is_proper_separator = False
                break

        if is_proper_separator:
            if self._separators_count[proposed_separator] < self._max_mult:
                self._separators_count[proposed_separator] += 1
                separator_recorded = True

        if self._ctl.get("debug", False):
            self._print_processed_separator(proposed_separator, separator_recorded)

    def _process_new_clique_gains(self, clq: frozenset):
        clique = tuple(clq)
        clique_size = len(clq)
        if clique_size < self._ctl["max_clique_size"]:
            facets = [clique]
        else:
            facets = list(itertools.combinations(clique, clique_size - 1))

        for facet in facets:
            for v in self._outstanding_nodes:
                gain, ranked_sep = self._gf(v, list(facet))
                heapq.heappush(self._gains_pq, (-gain, v, ranked_sep))

    def _logo(self, C, cliques, separators):
        J = np.zeros(C.shape)
        # For each clique, add the inverse of the submatrix defined by the clique indices.
        for clq in cliques:
            clqt = tuple(clq)
            J[np.ix_(clqt, clqt)] += LA.inv(C[np.ix_(clqt, clqt)])
        # For each separator, subtract the inverse of the submatrix defined by the separator indices.
        for sep, mult in (
            separators.items()
            if hasattr(separators, "items")
            else [(s, 1) for s in separators]
        ):
            if sep:  # ensure separator is non-empty
                sept = tuple(sep)
                J[np.ix_(sept, sept)] -= mult * LA.inv(C[np.ix_(sept, sept)])
        return J

    # Debug printing functions

    def _print_initial_state(self, first_cl):
        print(f'Seed Selection ({self._ctl["method"]})')
        print("  Seed clique:", format_frozenset(first_cl))
        print("  Selected based on gain function maximization")
        print("  Remaining nodes:", len(self._outstanding_nodes))
        print("---")

    def _print_added_clique(self, v, new_clique, parent_clique, sep):
        print("Iteration", self._iteration)
        print("  Added vertex:", v)
        print("  Proposed sub-clique:", format_frozenset(sep))
        print(
            "  Parent clique:",
            format_frozenset(parent_clique) if parent_clique else None,
        )
        print("  New clique:", format_frozenset(new_clique))

    def _print_processed_separator(self, proposed_separator, separator_recorded):
        minc = self._ctl.get("min_clique_size", 2)
        maxc = self._ctl.get("max_clique_size", 4)
        if len(proposed_separator) == 0:
            print("  → No separator recorded (empty proposed sub-clique)")
        elif not separator_recorded:
            if len(proposed_separator) >= maxc or len(proposed_separator) < (minc - 1):
                print(
                    f"  → Separator NOT recorded (size constraints: {len(proposed_separator)} not in [{minc - 1}, {maxc - 1}])"
                )
            else:
                print(
                    "  → Separator NOT recorded (proposed sub-clique equals existing clique - not proper)"
                )
        else:
            print(f"  → Separator RECORDED: {format_frozenset(proposed_separator)}")


class Gains:
    def __init__(self, C: np.ndarray, ctl: dict, gf_type: str = "sumsquares"):
        if gf_type == "sumsquares":
            self._W = np.square(C)
        else:
            # Modified to allow custom gain functions
            raise ValueError(f"Unknown gain function type: {gf_type}")
        self._threshold = ctl.get("threshold", 0.0)
        self._min_clique_size = ctl.get("min_clique_size", 1)
        self._max_clique_size = ctl.get("max_clique_size", 4)
        self._cachesize = ctl.get("cachesize", np.inf)

    def get_gain(self, v: int, sep: list[int]) -> tuple[float, frozenset]:
        values, ranked_sep = self._greedy_sortsep(v, sep)
        # Apply threshold and minimum clique size
        values, ranked_sep = self._apply_threshold(values, ranked_sep)
        gain = np.sum(values)
        return gain, frozenset(ranked_sep)

    def _greedy_sortsep(self, v: int, sep: list[int]) -> tuple[np.ndarray, np.ndarray]:
        cols = np.asarray(sep)
        weights = self._W[v, cols]
        order = np.argsort(weights)[::-1]  # descending

        values = weights[order]
        ranked_seps = cols[order]
        return values, ranked_seps

    def _apply_threshold(self, val, sep):
        # This function does the threshold on the individual edges
        idx = val >= self._threshold
        idx[0 : (self._min_clique_size - 1)] = True

        val = val[idx]
        sep = sep[idx]
        return val, sep
