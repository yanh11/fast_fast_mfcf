# Faster version
import numpy as np
import math
import itertools
from collections import Counter

##############################
# Utility Functions
##############################

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

# Start with one node
def first_clique(C, first=1):
    C1 = C.copy()
    r, c = np.nonzero(C <= C.mean())
    C1[r, c] = 0
    sums = C1.sum(axis=0)
    cand = np.argsort(-sums, kind='stable')  # stable sort
    clq = cand[:first]
    return clq

# Same starting clique as TMFG, this guarantees the same final graph as TMFG
def first_TMFG(C):
    W = np.square(C)
    flat_matrix = W.flatten()
    mean_flat_matrix = np.mean(flat_matrix)
    # Only count weights above the mean
    v = np.sum(W * (W > mean_flat_matrix), axis=1)
    sorted_v = np.argsort(v)[::-1]
    return sorted_v[:4]

def prune_gain_table(gt, threshold_nan=0.5):
    """
    Prune entries from the gain table when the fraction of disabled items exceeds threshold_nan.
    """
    n = len(gt.gains)
    if n == 0:
        return
    if np.mean(np.isnan(gt.gains)) > threshold_nan:
        new_nodes = []
        new_gains = []
        new_seps = []
        new_cliques = []
        for i in range(n):
            if not math.isnan(gt.gains[i]):
                new_nodes.append(gt.nodes[i])
                new_gains.append(gt.gains[i])
                new_seps.append(gt.separators[i])
                new_cliques.append(gt.cliques[i])
        gt.nodes, gt.gains, gt.separators, gt.cliques = new_nodes, new_gains, new_seps, new_cliques

##############################
# mfcf Algorithm and Helpers
##############################

def fast_mfcf(C, ctl, gain_function, cov_matrix=None):
    """
    Modified mfcf algorithm with performance improvements.
    
    Parameters:
    -----------
    C : np.ndarray
        Correlation matrix (used for clique/separator finding)
    ctl : dict
        Control parameters
    gain_function : callable
        Function to compute gains
    cov_matrix : np.ndarray, optional
        Covariance matrix (used for LoGo precision computation)
        If None, uses C for both
    
    Returns:
    --------
    cliques : list
        List of cliques
    separators : Counter
        Separator usage counts
    peo : list
        Perfect elimination order
    gt : gain_table
        Gain table
    J_logo : np.ndarray
        LoGo precision matrix
    """
    cliques = []
    separators = Counter()
    peo = []  # Perfect elimination order
    gt = gain_table()
    q, p = C.shape
    outstanding_nodes = list(range(p))
    
    # Choose the first clique based on the method.
    if ctl['method'] == 'TMFG':
        first_cl = first_TMFG(C)
        first_cl = frozenset(first_cl)
    elif ctl['method'] == 'MFCF':
        # For MFCF, start with the vertex that maximizes the gain function
        first_cl = first_clique(C)
        first_cl = frozenset(first_cl)
    else:
        raise ValueError("Unknown method: %s" % ctl['method'])
    
    # Debug output for seed step
    if ctl.get('debug', False):
        print(f'Seed Selection ({ctl["method"]})')
        print('  Seed clique:', format_frozenset(first_cl))
        print('  Selected based on gain function maximization')
        print('  Remaining nodes:', len([node for node in outstanding_nodes if node not in first_cl]))
        print('---')
    
    cliques.append(first_cl)
    outstanding_nodes = [node for node in outstanding_nodes if node not in first_cl]
    peo.extend(first_cl)
    
    # Initialize the gain table using the provided gain function.
    gt.nodes, gt.gains, gt.separators = gain_function(C, outstanding_nodes, first_cl, ctl)
    gt.separators = [frozenset(s) for s in gt.separators]
    gt.cliques = [first_cl] * len(gt.gains)
    
    # Use a Counter to track the number of times each separator is used.
    separator_counter = Counter(separators)
    
    iteration = 0
    while len(outstanding_nodes) > 0:
        iteration += 1
        # --- Coordination Number Check using the Counter ---
        for i, sep in enumerate(gt.separators):
            if (gt.gains[i] is None) or math.isnan(gt.gains[i]):
                continue
            s = frozenset(sep) if sep else frozenset()
            # length constraint
            minc = ctl.get('min_clique_size', 2)
            q = ctl.get('max_clique_size', 4)
            if not (len(s) >= (minc-1) and len(s) < q):
                gt.gains[i] = math.nan; continue
            # multiplicity constraint
            max_mult = ctl.get('max_separator_multiplicity', ctl.get('coordination_number', float('inf')))
            if s and (separators[s] >= max_mult):
                gt.gains[i] = math.nan; continue
            # subset-of-some-current-clique constraint
            valid_parent = any(s <= c for c in cliques)
            if not valid_parent:
                gt.gains[i] = math.nan

        
        the_gain = np.nanmax(gt.gains)
        if np.isnan(the_gain):
            # Fallback: choose the first outstanding node
            the_node = outstanding_nodes[0]
            the_sep = cliques[-1]
            parent_clique = cliques[-1]
            parent_clique_id = cliques.index(parent_clique)
            clique_extension = False
        elif the_gain < ctl['threshold']:
            # If gains do not meet threshold, start a new clique.
            the_node = outstanding_nodes[0]
            the_sep = frozenset()  # empty separator
            parent_clique = frozenset()
            parent_clique_id = None
            clique_extension = False
        else:
            idx = gt.gains.index(the_gain)
            the_node = gt.nodes[idx]
            the_sep = frozenset(gt.separators[idx])
            parent_clique = None
            for _c in cliques:
                if the_sep <= _c:
                    parent_clique = _c
                    break
            try:
                parent_clique_id = cliques.index(parent_clique) if parent_clique is not None else None
            except ValueError:
                parent_clique_id = None
            clique_extension = True
        
#         print("Iteration:", iteration, "Selected node:", the_node, "with separator:", the_sep, "and gain:", the_gain)
        
        # Snapshot cliques before modification for local separator computation
        cliques_before = list(cliques)
        # Form the new clique.
        new_clique = frozenset(set(the_sep).union({the_node}))
        peo.append(the_node)
        outstanding_nodes.remove(the_node)
        if ctl.get('debug', False):
            print('Iteration', iteration)
            print('  Added vertex:', the_node)
            print('  Proposed sub-clique:', format_frozenset(the_sep))
            print('  Parent clique:', format_frozenset(parent_clique) if parent_clique else None)
            print('  New clique:', format_frozenset(new_clique))

        # --- NEW: keep only maximal cliques (remove any proper subset of the new one) ---
        if len(new_clique) > 1:  # only meaningful if it actually grew
            # Collect proper subset cliques to drop (strict subset)
            to_remove = [c for c in cliques if c < new_clique]
            for c in to_remove:
                cliques.remove(c)

        # Previous conditional logic simplified: always append the new (possibly maximal) clique
        cliques.append(new_clique)

        # CORRECTED: MFCF separator logic - use proposed sub-clique as separator
        # The separator is simply the proposed sub-clique, unless it equals an entire existing clique
        proposed_separator = frozenset(the_sep)
        separator_recorded = False
        
        # Check if proposed sub-clique is a valid separator
        if len(proposed_separator) > 0:  # Non-empty
            # Check size constraints
            minc = ctl.get('min_clique_size', 2)
            q = ctl.get('max_clique_size', 4)
            
            if (len(proposed_separator) >= (minc-1)) and (len(proposed_separator) < q):
                # Check if proposed sub-clique is NOT equal to any entire existing clique
                is_proper_separator = True
                for existing_clique in cliques_before:  # Check against cliques before adding new one
                    if proposed_separator >= existing_clique:  # Not a proper subset
                        is_proper_separator = False
                        break
                
                if is_proper_separator:
                    max_mult = ctl.get('max_separator_multiplicity', ctl.get('coordination_number', float('inf')))
                    if separators[proposed_separator] < max_mult:
                        separators[proposed_separator] += 1
                        separator_recorded = True
        
        if ctl.get('debug', False):
            if len(proposed_separator) == 0:
                print('  → No separator recorded (empty proposed sub-clique)')
            elif not separator_recorded:
                if len(proposed_separator) >= q or len(proposed_separator) < (minc-1):
                    print(f'  → Separator NOT recorded (size constraints: {len(proposed_separator)} not in [{minc-1}, {q-1}])')
                else:
                    print('  → Separator NOT recorded (proposed sub-clique equals existing clique - not proper)')
            else:
                print(f'  → Separator RECORDED: {format_frozenset(proposed_separator)}')
        
        if len(outstanding_nodes) == 0:
            break
        
        # Update the gain table for the new clique.
        nodes_new, gains_new, seps_new = gain_function(C, outstanding_nodes, new_clique, ctl)
        seps_new = [frozenset(s) for s in seps_new]
        add_cliques = [new_clique] * len(gains_new)
        gt.nodes.extend(nodes_new)
        gt.gains.extend(gains_new)
        gt.separators.extend(seps_new)
        gt.cliques.extend(add_cliques)
        
        # --- Vectorized Update: Disable any gain records that refer to the newly added node ---
        nodes_arr = np.array(gt.nodes)
        gains_arr = np.array(gt.gains,dtype=float)
        gains_arr[nodes_arr == the_node] = math.nan
        
        # If drop_sep is enabled, disable candidates with the used separator.
        if ctl['drop_sep'] == True:
            mask = np.array([sep == the_sep for sep in gt.separators])
            gains_arr[mask] = math.nan
        
        gt.gains = gains_arr.tolist()
        
        # Debug: Show updated gain table state AFTER all updates
        if ctl.get('debug', False):
            # Show gain table state (potential sub-cliques) after updates
            valid_entries = [(gt.nodes[i], format_frozenset(gt.separators[i]), gt.gains[i]) 
                           for i in range(len(gt.gains)) 
                           if not (math.isnan(gt.gains[i]) if gt.gains[i] is not None else True)]
            # Sort by gain value in descending order (largest gains first)
            valid_entries.sort(key=lambda x: x[2], reverse=True)
            print(f'  Updated gain table: {len(valid_entries)} valid entries from {len(gt.gains)} total (sorted by largest gains first)')
            for node, subclique, gain in valid_entries[:5]:  # Show first 5
                print(f'    Node {node} + sub-clique {subclique} = gain {gain:.4f}')
            if len(valid_entries) > 5:
                print(f'    ... and {len(valid_entries)-5} more')
            print('---')
        
        # Optionally prune the gain table periodically
        prune_gain_table(gt, threshold_nan=0.5)
    
    # Compute LoGo precision matrix using the appropriate matrix
    matrix_for_logo = cov_matrix if cov_matrix is not None else C
    J_logo = logo(matrix_for_logo, cliques, separators)
    
    return cliques, separators, peo, gt, J_logo

def mfcf_control():
    ctl = {
        'min_clique_size': 1, 
        'max_clique_size': 4, 
        'coordination_number': np.inf,  # maximum allowed uses of a separator
        'cachesize': np.inf,
        'threshold': 0.00,
        'drop_sep': True, 
        'method': 'MFCF'
    }
    return ctl

def logo(C, cliques, separators):
    import numpy.linalg as LA
    J = np.zeros(C.shape)
    # For each clique, add the inverse of the submatrix defined by the clique indices.
    for clq in cliques:
        clqt = tuple(clq)
        J[np.ix_(clqt, clqt)] += LA.inv(C[np.ix_(clqt, clqt)])
    # For each separator, subtract the inverse of the submatrix defined by the separator indices.
    for sep, mult in (separators.items() if hasattr(separators, 'items') else [(s,1) for s in separators]):
        if sep:  # ensure separator is non-empty
            sept = tuple(sep)
            J[np.ix_(sept, sept)] -= mult * LA.inv(C[np.ix_(sept, sept)])
    return J


import itertools as itertools
import numpy as np

def sumsquares_gen(M, v, clq, ct_control):
    """
    M similarity matrix
    v vector of outstanding nodes
    clq clique
    ct_control parameters for the clique expansion algorithm
    """
    nodes = []
    gains = []
    seps = []
    
    clq = tuple(clq)
    csz = len(clq)
    vn = len(v)
    W = M*M
    
    if 'threshold' in ct_control:
        threshold = ct_control['threshold']
    else:
        threshold = 0.0
        
    if 'min_clique_size' in ct_control:
        min_clique_size = ct_control['min_clique_size']
    else:
        min_clique_size = 1
    
    if 'max_clique_size' in ct_control:
        max_clique_size = ct_control['max_clique_size']
    else:
        max_clique_size = 4
    
    if 'cachesize' in ct_control:
        cachesize = ct_control['cachesize']
    else:
        cachesize = np.inf # we can lower this to trade for computation speed
     
    if 'coordination_number' in ct_control:
        coordination_number = ct_control['coordination_number']
    else:
        coordination_number = np.inf
        
    if 'method' in ct_control:
        method = ct_control['method']
    else:
        method = 'MFCF' # we start with one node that has the highest column sum by default
        
    if csz < max_clique_size:
        facets = list()
        facets.append(clq)
    else:
        facets = list(itertools.combinations(clq, csz-1))
    
    block_rows = len(facets)
    ncol = len(facets[0])
    
    the_vs = np.sort(np.tile(v, block_rows)) # nodes in order
    the_fs = facets * vn # facets as they are generated
    
    ranked_values, ranked_seps = greedy_sortsep_v(the_vs, the_fs, W)
    ranked_values_thr, ranked_seps_thr =  apply_threshold_v(ranked_values, ranked_seps, min_clique_size, threshold)
    
    gains = list(map(sum, ranked_values_thr))
    
    selector = np.tile( list(range(1,block_rows+1)), vn)
    
    the_table = list(zip(the_vs, gains, ranked_seps_thr))
    
    idx = np.where(selector<=cachesize)[0].tolist()
    
    the_table = [the_table[x] for x in idx]
    
    nodes = [x[0] for x in the_table]
    gains = [x[1] for x in the_table]
    seps =  [frozenset(x[2].tolist()) for x in the_table]
    
    return nodes, gains, seps

def greedy_sortsep(vtx, sep, W):
    
    w = W[vtx, sep]
    sepv = np.array(sep)
    sep_ranked = np.argsort(w)[::-1]
    values = w[sep_ranked]
    sep_ranked = sepv[sep_ranked]
    
    return values, sep_ranked

def greedy_sortsep_v(vertices, sets, W):
    
    local_fun = lambda x,y: greedy_sortsep(x, y, W)
    retval = list(map(local_fun, vertices, sets))
    ranked_values = [x[0] for x in retval]
    ranked_seps = [x[1] for x in retval]
    
    return ranked_values, ranked_seps

def apply_threshold(val, sep, mincsize, threshold):
    # This function does the threshold on the individual edges
    idx = val >= threshold
    idx[0:(mincsize-1)] = True
    val=val[idx]
    sep=sep[idx]
    
    return val, sep

def apply_threshold_v(ranked_values, ranked_seps, mincsize, threshold):
    
    local_fun= lambda x, y: apply_threshold(x, y, mincsize, threshold)
    retval = list(map(local_fun, ranked_values, ranked_seps))
    ranked_values = [x[0] for x in retval]
    ranked_seps = [x[1] for x in retval]
    
    return ranked_values, ranked_seps

class gain_table:
    
    def __init__(self):
        self.nodes = list()
        self.cliques = list()
        self.separators = list()
        self.coordination_numbers = list()
        self.gains = list()
        return