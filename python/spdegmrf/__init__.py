"""Reference implementation of the SPDE-GMRF estimation and forward-filter pipeline.

Modules
-------
mesh        structured triangular mesh, point location, barycentric weights
fem         P1 finite-element assembly of the lumped mass matrix C and stiffness G
cholesky    sparse Cholesky with a reusable symbolic analysis
spde        precision Q(theta) = tau^2 (kappa^4 C + 2 kappa^2 G + G C^-1 G)
projector   sparse projector P from mesh nodes to sensor locations
estimate    Gaussian marginal likelihood, posterior field and (kappa, tau) estimation
lattice     the 20 by 20 lattice ordering experiment (fill-in counts)
hmm         float forward filter over the four behavioural regimes
quant       int8 quantisation and the bit-exact integer forward filter mirror
export      C header export for the firmware
"""

__all__ = [
    "mesh", "fem", "cholesky", "spde", "projector", "estimate",
    "lattice", "hmm", "quant", "export",
]
