import heapq
import itertools
import logging
from collections import Counter
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import numpy.linalg as LA

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# Example usage from caller:
# logging.basicConfig(level=logging.INFO)  # or DEBUG


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Node = int
Clique = FrozenSet[int]
Separator = FrozenSet[int]


# -----------------------------------------------------------------------------
# Helper formatting (debug/pretty-print utilities)
# -----------------------------------------------------------------------------
def format_frozenset(fs: Iterable[int]) -> str:
    """Convert a frozenset (or any iterable) to readable, sorted list format."""
    try:
        return str(sorted(int(x) for x in fs)) if fs else "[]"
    except (ValueError, TypeError):
        return str(sorted(list(fs))) if fs else "[]"


def format_cliques_list(cliques: Iterable[Clique]) -> List[str]:
    """Format list of cliques to readable strings."""
    return [format_frozenset(clq) for clq in cliques]


def format_counter(counter: Counter) -> str:
    """Format Counter with frozenset keys to readable format."""
    if not counter:
        return "{}"
    formatted: Dict[str, int] = {}
    for fs, count in counter.items():
        key = format_frozenset(fs) if fs else "[]"
        formatted[key] = count
    return str(formatted)


# -----------------------------------------------------------------------------
# Default control
# -----------------------------------------------------------------------------
def mfcf_control() -> Dict[str, object]:
    """Return default control dictionary for MFCF/TMFG runs."""
    return {
        "min_clique_size": 1,
        "max_clique_size": 4,
        "coordination_number": np.inf,  # maximum allowed uses of a separator
        "cachesize": np.inf,
        "threshold": 0.00,
        "drop_sep": True,
        "method": "MFCF"
    }


# =============================================================================
# Gains
# =============================================================================
class Gains:
    """Gain function handler (currently supports 'sumsquares')."""

    def __init__(self, C: np.ndarray, ctl: Dict, gf_type: str = "sumsquares"):
        if gf_type == "sumsquares":
            self._W = np.square(C)
        else:
            raise ValueError(f"Unknown gain function type: {gf_type}")

        self._threshold: float = float(ctl.get("threshold", 0.0))
        self._min_clique_size: int = int(ctl.get("min_clique_size", 1))
        self._max_clique_size: int = int(ctl.get("max_clique_size", 4))
        self._cachesize = ctl.get("cachesize", np.inf)

    def get_gain(self, v: Node, sep: List[Node]) -> Tuple[float, Separator]:
        """Compute gain for adding v with a candidate separator list."""
        values, ranked_sep = self._greedy_sortsep(v, sep)
        values, ranked_sep = self._apply_threshold(values, ranked_sep)
        gain = float(np.sum(values))
        return gain, frozenset(int(x) for x in ranked_sep)

    # --- internals ------------------------------------------------------------

    def _greedy_sortsep(self, v: Node, sep: List[Node]) -> Tuple[np.ndarray, np.ndarray]:
        """Sort separator nodes for v by descending weight."""
        cols = np.asarray(sep, dtype=int)
        weights = self._W[v, cols]
        order = np.argsort(weights)[::-1]
        return weights[order], cols[order]

    def _apply_threshold(self, val: np.ndarray, sep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply per-edge threshold; always keep first (min_clique_size-1) edges."""
        idx = val >= self._threshold
        # ensure enough seeds for minimum clique size
        keep_prefix = max(0, self._min_clique_size - 1)
        idx[:keep_prefix] = True
        return val[idx], sep[idx]


# =============================================================================
# MFCF
# =============================================================================
class MFCF:
    """Maximum Filtering Clique Forest (MFCF) builder."""

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        pass

    def fast_mfcf(
        self,
        C: np.ndarray,
        ctl: Dict,
        gf_type: str,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[List[Clique], Counter, List[Node], np.ndarray]:
        """
        Run the MFCF/TMFG process and return:
          cliques, separators_count, perfect_elimination_order (peo), J_logo
        """
        self._C = C
        self._ctl = ctl
        self._gf = Gains(C, ctl, gf_type).get_gain

        self._initialise()
        self._compute_mfcf()

        matrix_for_logo = cov_matrix if cov_matrix is not None else C
        J_logo = self._logo(matrix_for_logo, self._cliques, self._separators_count)
        return self._cliques, self._separators_count, self._peo, J_logo

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------
    def _initialise(self) -> None:
        """Prepare data structures and seed clique."""
        self._gains_pq: List[Tuple[float, Node, Separator]] = []
        self._iteration = 0
        self._max_mult = self._ctl.get(
            "max_separator_multiplicity",
            self._ctl.get("coordination_number", float("inf")),
        )

        first_cl = self._get_first_clique()

        self._cliques: List[Clique] = [first_cl]
        self._separators_count: Counter = Counter()
        self._seen_separators: set[Separator] = set()
        self._peo: List[Node] = [v for v in first_cl]  # Perfect elimination order
        self._outstanding_nodes: List[Node] = [
            v for v in range(self._C.shape[0]) if v not in first_cl
        ]

        self._log_initial_state(first_cl)

        self._process_new_clique_gains(first_cl)

    def _get_first_clique(self) -> Clique:
        """Choose first clique according to ctl['method']."""
        method = self._ctl["method"]
        if method == "TMFG":
            return frozenset(self._first_TMFG())
        if method == "MFCF":
            return frozenset(self._first_clique())
        raise ValueError(f"Unknown method: {method}")

    def _first_clique(self, first: int = 1) -> np.ndarray:
        """Seed with node(s) having high sum of weights above-mean edges."""
        C1 = self._C.copy()
        r, c = np.nonzero(self._C <= self._C.mean())
        C1[r, c] = 0
        sums = C1.sum(axis=0)
        cand = np.argsort(-sums, kind="stable")
        return cand[:first]

    def _first_TMFG(self) -> np.ndarray:
        """TMFG-compatible 4-node seed."""
        W = np.square(self._C)
        mean_flat = float(np.mean(W))
        v = np.sum(W * (W > mean_flat), axis=1)
        sorted_v = np.argsort(v)[::-1]
        return sorted_v[:4]

    # -------------------------------------------------------------------------
    # Main algorithm loop
    # -------------------------------------------------------------------------
    def _compute_mfcf(self) -> None:
        """Greedy loop popping best (gain, v, sep) and updating structures."""
        while self._outstanding_nodes:
            self._iteration += 1
            gain, v, sep = heapq.heappop(self._gains_pq)
            if self._should_skip_candidate(gain, v, sep):
                continue

            v, sep, parent_clique = self._apply_threshold_and_find_parent(gain, v, sep)
            cliques_before = list(self._cliques)
            new_clique = self._add_new_clique(parent_clique, sep, v)

            self._check_proposed_separator(sep, cliques_before)

            if not self._outstanding_nodes:
                break
            self._process_new_clique_gains(new_clique)

    # -------------------------------------------------------------------------
    # Candidate checks & parent search
    # -------------------------------------------------------------------------
    def _should_skip_candidate(self, gain: float, v: Node, sep: Separator) -> bool:
        """Filter heap candidates by availability, size, multiplicity, and validity."""
        if np.isnan(gain) or v not in self._outstanding_nodes:
            return True
        # If drop_sep is enabled, disable candidates with a seen/used separator.
        if self._ctl.get("drop_sep", False):
            if sep in self._seen_separators or self._separators_count[sep] > 0:
                return True
        # length constraint
        minc = int(self._ctl.get("min_clique_size", 2))
        maxc = int(self._ctl.get("max_clique_size", 4))
        if not (len(sep) >= minc - 1 and len(sep) < maxc):
            return True
        # multiplicity constraint
        if self._separators_count[sep] >= self._max_mult:
            return True
        # subset-of-some-current-clique constraint
        if not any(sep.issubset(clq) for clq in self._cliques):
            return True
        return False

    def _apply_threshold_and_find_parent(
        self, gain: float, v: Node, sep: Separator
    ) -> Tuple[Node, Separator, Optional[Clique]]:
        """Decide if we start a new component (below threshold) or attach to a parent clique."""
        pos_gain = -gain  # negate back to positive for threshold compare
        if pos_gain < self._ctl["threshold"]:
            # start a new clique
            v = self._outstanding_nodes[0]
            sep = frozenset()
            parent_clique = frozenset()
        else:
            parent_clique = self._find_parent_clique_for_separator(sep)
        return v, sep, parent_clique

    def _find_parent_clique_for_separator(self, sep: Separator) -> Optional[Clique]:
        """Find a current clique that contains sep."""
        for clq in self._cliques:
            if sep <= clq:
                return clq
        return None

    # -------------------------------------------------------------------------
    # Clique/separator updates
    # -------------------------------------------------------------------------
    def _add_new_clique(
        self, parent_clique: Optional[Clique], sep: Separator, v: Node
    ) -> Clique:
        """Add new clique, keep only maximal cliques, and update PEO/outstanding."""
        new_clique: Clique = frozenset(sep | {v})
        self._peo.append(v)
        self._outstanding_nodes.remove(v)

        self._log_added_clique(v, new_clique, parent_clique, sep)

        # keep only maximal cliques (drop strict subsets of the new one)
        if len(new_clique) > 1:
            to_remove = [c for c in self._cliques if c < new_clique]
            for c in to_remove:
                self._cliques.remove(c)

        self._cliques.append(new_clique)
        return new_clique

    def _check_proposed_separator(self, proposed_separator: Separator, cliques_before: List[Clique]) -> None:
        """
        Record separator if it's a proper subset of an existing clique
        and passes size/multiplicity constraints.
        """
        if not proposed_separator:
            return  # Empty separator, nothing to record

        self._seen_separators.add(proposed_separator)
        minc = int(self._ctl.get("min_clique_size", 2))
        maxc = int(self._ctl.get("max_clique_size", 4))
        sep_len = len(proposed_separator)
        if not (sep_len >= minc - 1 and sep_len < maxc):
            return

        # proper subset check against cliques BEFORE adding the new one
        is_proper_separator = True
        for existing_clique in cliques_before:
            if proposed_separator >= existing_clique:
                is_proper_separator = False
                break

        separator_recorded = False
        if is_proper_separator and self._separators_count[proposed_separator] < self._max_mult:
            self._separators_count[proposed_separator] += 1
            separator_recorded = True

        self._log_processed_separator(proposed_separator, separator_recorded)

    def _process_new_clique_gains(self, clq: Clique) -> None:
        """Push gain candidates for all facets of the clique vs outstanding nodes."""
        clique = tuple(clq)
        clique_size = len(clq)
        max_size = int(self._ctl["max_clique_size"])

        facets = [clique] if clique_size < max_size else list(itertools.combinations(clique, clique_size - 1))
        for facet in facets:
            for v in list(self._outstanding_nodes):
                gain, ranked_sep = self._gf(v, list(facet))
                heapq.heappush(self._gains_pq, (-gain, v, ranked_sep))

    # -------------------------------------------------------------------------
    # logo computation
    # -------------------------------------------------------------------------
    def _logo(self, C: np.ndarray, cliques: List[Clique], separators: Counter) -> np.ndarray:
        """Compute sparse inverse estimator via cliques minus separators."""
        J = np.zeros(C.shape)
        # For each clique, add the inverse of the submatrix defined by the clique indices.
        for clq in cliques:
            clqt = tuple(clq)
            J[np.ix_(clqt, clqt)] += LA.inv(C[np.ix_(clqt, clqt)])

        # For each separator, subtract the inverse of the submatrix defined by the separator indices.
        for sep, mult in separators.items():
            if sep:  # non-empty
                sept = tuple(sep)
                J[np.ix_(sept, sept)] -= mult * LA.inv(C[np.ix_(sept, sept)])

        return J

    # -------------------------------------------------------------------------
    # Debug logging
    # -------------------------------------------------------------------------
    def _log_initial_state(self, first_cl: Clique) -> None:
        logger.info("Seed Selection (%s)", self._ctl["method"])
        logger.info("  Seed clique: %s", format_frozenset(first_cl))
        logger.info("  Selected based on gain function maximization")
        logger.info("  Remaining nodes: %d", len(self._outstanding_nodes))
        logger.info("---")

    def _log_added_clique(
        self, v: Node, new_clique: Clique, parent_clique: Optional[Clique], sep: Separator
    ) -> None:
        logger.info("Iteration %d", self._iteration)
        logger.info("  Added vertex: %s", v)
        logger.info("  Proposed sub-clique: %s", format_frozenset(sep))
        logger.info("  Parent clique: %s", format_frozenset(parent_clique) if parent_clique else None)
        logger.info("  New clique: %s", format_frozenset(new_clique))

    def _log_processed_separator(self, proposed_separator: Separator, separator_recorded: bool) -> None:
        minc = int(self._ctl.get("min_clique_size", 2))
        maxc = int(self._ctl.get("max_clique_size", 4))
        if len(proposed_separator) == 0:
            logger.info("  → No separator recorded (empty proposed sub-clique)")
        elif not separator_recorded:
            if len(proposed_separator) >= maxc or len(proposed_separator) < (minc - 1):
                logger.info(
                    "  → Separator NOT recorded (size constraints: %d not in [%d, %d])",
                    len(proposed_separator), minc - 1, maxc - 1
                )
            else:
                logger.info("  → Separator NOT recorded (proposed sub-clique equals existing clique - not proper)")
        else:
            logger.info("  → Separator RECORDED: %s", format_frozenset(proposed_separator))
