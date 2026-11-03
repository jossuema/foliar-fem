"""
P1 FEM core on the triangular mesh (valid in R^2 and in R^3).

Contents:
  - areas and shape-function gradients
  - stiffness matrix (cotangent Laplacian) and mass matrix
  - geodesic distances via the heat method of Crane et al. (2013)

The heat method is the underlying reason why meshing is not decorative here:
solving for the geodesic literally goes through assembling and solving two FEM
systems on the mesh.

    1)  (M + t S) u = delta_source        heat diffusion, short time
    2)  X = -grad(u) / |grad(u)|          unit direction field
    3)  S phi = div(X)                    Poisson; phi is the distance
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------
# Element geometry
# --------------------------------------------------------------------------
def _as3d(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, float)
    if V.shape[1] == 2:
        V = np.column_stack([V, np.zeros(len(V))])
    return V


def triangle_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    V = _as3d(V)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def shape_gradients(V: np.ndarray, F: np.ndarray):
    """
    Gradients of the P1 hat functions, per triangle.

    grad(phi_a) = (N x e_a) / (2A), with e_a the edge opposite vertex a.
    Returns G of shape (m, 3, 3): G[t, a] = grad(phi_a) on triangle t.
    """
    V = _as3d(V)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    twoA = np.linalg.norm(n, axis=1)
    A = 0.5 * twoA
    N = n / np.maximum(twoA, 1e-300)[:, None]

    e = np.stack([v2 - v1, v0 - v2, v1 - v0], axis=1)     # e[:, a] opposite a
    G = np.cross(N[:, None, :], e) / np.maximum(twoA, 1e-300)[:, None, None]
    return G, A


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
def stiffness_matrix(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """P1 stiffness: S_ab = sum_T A_T grad(phi_a).grad(phi_b). Positive semidefinite."""
    G, A = shape_gradients(V, F)
    n = len(V)
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            rows.append(F[:, a])
            cols.append(F[:, b])
            vals.append(A * np.einsum("ti,ti->t", G[:, a], G[:, b]))
    S = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    return S


def mass_matrix(V: np.ndarray, F: np.ndarray, lumped: bool = False):
    """P1 mass. Consistent: A/12 * [[2,1,1],[1,2,1],[1,1,2]]. Lumped: A/3."""
    A = triangle_areas(V, F)
    n = len(V)
    if lumped:
        d = np.zeros(n)
        np.add.at(d, F.ravel(), np.repeat(A / 3.0, 3))
        return sp.diags(d).tocsr()
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            w = (2.0 if a == b else 1.0) / 12.0
            rows.append(F[:, a])
            cols.append(F[:, b])
            vals.append(w * A)
    return sp.coo_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n)).tocsr()


def surface_area(V: np.ndarray, F: np.ndarray, subset: np.ndarray | None = None) -> float:
    """Mesh area by quadrature. On the lifted mesh this is the true 3D area."""
    A = triangle_areas(V, F)
    return float(A.sum() if subset is None else A[subset].sum())


def components(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Connected components of the edge graph (a torn leaf can split)."""
    n = len(V)
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    G = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, labels = csgraph.connected_components(G, directed=False)
    return labels


# --------------------------------------------------------------------------
# Geodesics: heat method
# --------------------------------------------------------------------------
def geodesic_distance(V: np.ndarray, F: np.ndarray, source_nodes,
                      m: float = 1.0) -> np.ndarray:
    """
    Geodesic distance from a set of source nodes, over the mesh surface.
    Returns nan on components that contain no source (a strip torn loose by the
    wind, for instance).

    MEASURED LIMITATION (see validate.py). Step 1 diffuses heat for a time
    t = m*h^2 and the solution decays as exp(-d^2/4t). On a large domain the far
    field UNDERFLOWS to exact zero, and there is no gradient direction left to
    recover there: the information was destroyed, not degraded.

    The underflow horizon sits at d ~ sqrt(4*700*m)*h. With m = 0.10 that falls
    at ~18 cm on a leaf with distances up to 36.6 cm, so more than half the
    domain ends up on the wrong side. The optimum is where the horizon just
    exceeds the domain diameter; above it, oversmoothing starts to lengthen the
    distances. Measured on the reference mesh:

        m         0.10   0.20   0.30   0.40   0.50   0.80   1.00
        RMSE cm   1.615  0.996  0.610  0.654  0.808  1.257  1.509

    The optimum is m = 0.30 with RMSE 0.610 cm. Even so it loses by a factor 7
    against Dijkstra on the 2-ring (RMSE 0.084 cm), so the default method in
    fem.geodesic() is 'dijkstra2' and this one stays as a documented
    alternative, useful if the computation ever moves to non-trivial topology.

    NOTE: an earlier version of this note blamed the failure on a "noise floor
    of the direct solver (~1e-11)". That is false: the measured relative
    residual of the solve ranges from 3.4e-16 to 1.5e-15. The mechanism is
    underflow of the data, not solver error.
    """
    V = _as3d(V)
    n = len(V)
    source = np.zeros(n, bool)
    source[np.asarray(source_nodes, int)] = True
    if not source.any():
        raise ValueError("no source nodes")

    S = stiffness_matrix(V, F)
    M = mass_matrix(V, F, lumped=True)
    G, A = shape_gradients(V, F)

    edges = np.linalg.norm(V[F[:, [1, 2, 0]]] - V[F[:, [0, 1, 2]]], axis=2)
    h = float(edges.mean())
    t = m * h * h

    labels = components(V, F)
    phi = np.full(n, np.nan)

    for c in np.unique(labels):
        nodes = np.flatnonzero(labels == c)
        if not source[nodes].any():
            continue                                   # detached fragment
        tri = np.flatnonzero(np.isin(F[:, 0], nodes))
        loc = -np.ones(n, int)
        loc[nodes] = np.arange(len(nodes))

        Sc = S[nodes][:, nodes].tocsc()
        Mc = M[nodes][:, nodes].tocsc()
        src = source[nodes]

        # --- step 1: heat diffusion ---
        u = spla.spsolve((Mc + t * Sc).tocsc(), src.astype(float))

        # --- step 2: direction field ---
        # CAREFUL: u decays as exp(-d^2/4t), so far from the source it is of
        # order 1e-200. Only the DIRECTION of the gradient matters, so the
        # threshold for discarding a triangle has to be RELATIVE. An absolute
        # threshold (1e-14, say) zeroes the field across the whole far field and
        # the distance stops growing.
        Ft = F[tri]
        Gt = G[tri]
        gu = np.einsum("ta,tai->ti", u[loc[Ft]], Gt)
        nrm = np.linalg.norm(gu, axis=1)
        X = -gu / np.maximum(nrm, 1e-300)[:, None]
        X[nrm <= 0.0] = 0.0                     # exact zero only (underflow)

        # --- step 3: Poisson S phi = div X ---
        b = np.zeros(len(nodes))
        for a in range(3):
            np.add.at(b, loc[Ft[:, a]], A[tri] * np.einsum("ti,ti->t", Gt[:, a], X))

        # Pure Neumann: S is singular; one degree of freedom is pinned at the source.
        pin = int(np.flatnonzero(src)[0])
        keep = np.ones(len(nodes), bool)
        keep[pin] = False
        Sr = Sc[keep][:, keep].tocsc()
        x = spla.spsolve(Sr, b[keep])

        p = np.zeros(len(nodes))
        p[keep] = x
        p -= p[src].min()
        phi[nodes] = p

    return phi


def _two_ring_links(i: np.ndarray, j: np.ndarray, n: int, block: int = 20_000):
    """
    Vertex pairs at graph distance <= 2 on the edge graph, i.e. the pattern of A^2.

    CAREFUL: these are ALL pairs at distance 2, not just the edge-flip links. In
    particular the pattern also includes ADJACENT pairs, because two neighbouring
    vertices always have a length-2 path through the third vertex of a shared
    triangle. Callers have to deduplicate against the edge list: otherwise
    coo_matrix sums the duplicates and every real edge ends up weighing double.
    That bug has already been made once.

    It is computed in row blocks: on a pathological mesh (over-simplified
    contours make Ruppert's refiner explode) a single A^2 can ask for hundreds
    of GB, and that mesh still has to be processable.
    """
    A = sp.coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(n, n)).tocsr()
    A = (A + A.T).astype(bool).astype(np.int8)
    rows, cols = [], []
    for s in range(0, n, block):
        P = (A[s:s + block] @ A).tocoo()
        rows.append(P.row + s)
        cols.append(P.col)
    return np.concatenate(rows), np.concatenate(cols)


def dijkstra_distance(V: np.ndarray, F: np.ndarray, source_nodes,
                      ring: int = 2) -> np.ndarray:
    """
    Geodesic by shortest paths on the mesh graph.

    ring=1 uses the edges only: the path is forced to break at every vertex and
    the bias is ~+3.5% upward. ring=2 adds the pairs at graph distance 2, which
    admits diagonal crossings and brings the bias down to ~+0.2% at O(m) cost
    and bounded degree.

    The pairs are DEDUPLICATED before assembly. This is not a detail: A^2 also
    contains the adjacent pairs, and coo_matrix sums duplicates, so without
    deduplication every real mesh edge would weigh twice its length while the
    2-ring links weighed correctly. Measured before the fix: 63,862 edges with
    weight exactly 2x, and the error against the reference eikonal field was
    +0.63% with RMSE 0.180 cm instead of +0.22% with RMSE 0.084 cm.

    KNOWN LIMITATION: the metrication bias of graph Dijkstra does NOT converge
    to zero under refinement, because the angular resolution of the stencil is
    scale invariant. Measured over a 6.5x range in triangle count, the error
    stays at ~+0.2% (ring=2) and ~+3.5% (ring=1). It is a floor, not a
    resolution-limited error: refining the mesh does not improve it.

    It is an independent control on the heat method: it shares no code with it.
    """
    V = _as3d(V)
    n = len(V)
    src = np.asarray(source_nodes, int).ravel()
    if src.size == 0:
        raise ValueError("no source nodes: the geodesic field would come out all nan "
                         "and downstream that turns into a weighted index of 0.0 "
                         "with the appearance of data")
    if src.min() < 0 or src.max() >= n:
        raise ValueError(f"source nodes out of range [0, {n})")
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    if ring >= 2:
        a, b = _two_ring_links(i, j, n)
        i = np.concatenate([i, a])
        j = np.concatenate([j, b])

    # deduplicate the unordered pair: each edge appears several times (once per
    # incident triangle, plus once per length-2 path that duplicates it)
    ij = np.unique(np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1), axis=0)
    i, j = ij[:, 0], ij[:, 1]

    w = np.linalg.norm(V[i] - V[j], axis=1)
    keep = w > 0
    G = sp.coo_matrix((w[keep], (i[keep], j[keep])), shape=(n, n))
    G = G.maximum(G.T).tocsr()
    d = csgraph.dijkstra(G, directed=False, indices=src, min_only=True)
    d[np.isinf(d)] = np.nan
    return d


def interpolate_to_triangles(values: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Nodal average -> per-triangle value."""
    return np.nanmean(values[F], axis=1)


def boundary_edges(F: np.ndarray) -> np.ndarray:
    """Edges belonging to a single triangle."""
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def boundary_length(V: np.ndarray, F: np.ndarray) -> float:
    """Mesh perimeter. On the lifted mesh this is the true 3D perimeter."""
    V = _as3d(V)
    e = boundary_edges(F)
    return float(np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1).sum())


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
def geodesic(V, F, source_nodes, method: str = "dijkstra2", m: float = 0.30):
    """
    Geodesic distance to the source set.

    Measured against the reference eikonal field on the leaf (see validate.py):

        method                 error in the mean    RMSE
        dijkstra2 (default)          +0.22 %       0.084 cm
        dijkstra                     +3.47 %       0.721 cm
        heat (m = 0.30)              +1.92 %       0.610 cm

    method:
      'dijkstra2' (default) - shortest paths on the 2-ring graph. The most
                  accurate and the fastest of the three on this prototype's
                  meshes. Its residual bias does NOT drop under refinement: it
                  is a metrication floor (see dijkstra_distance).
      'dijkstra'  - edges only. Metrication bias of ~+3.5%.
      'heat'      - Crane's heat method. Solves two FEM systems on the mesh.
                  Elegant, and the one that justifies the Laplacian, but on
                  large domains its accuracy is limited by underflow in the
                  diffusion step (see the note in geodesic_distance).

    m : for 'heat' only. Sets t = m*h^2. The default 0.30 is the MEASURED
        optimum on the reference mesh, not a theoretical value.
    """
    if method == "heat":
        return geodesic_distance(V, F, source_nodes, m=m)
    if method == "dijkstra":
        return dijkstra_distance(V, F, source_nodes, ring=1)
    if method == "dijkstra2":
        return dijkstra_distance(V, F, source_nodes, ring=2)
    raise ValueError(f"unknown method: {method}")
