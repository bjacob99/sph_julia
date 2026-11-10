```@meta
CurrentModule = LagrangianStochasticConvection
```

# Kernel-mean algorithm

## Philosophy

One of the strengths of particle methods is that they allow to extract mean flow variables directly from the set of notional particles used to represent the flow. Averages of the appropriate particle-based quantity over locally filtered particles yield the requested local ensemble average. The filtering is done through a *kernel estimation* scheme of the form
```math
\overline{Q}(\mathbf{x}, t) = \sum_{k=1}^N \mathcal{V}^{\ast (k)} Q^{\ast (k)} K \left( \mathbf{x}^{\ast (k)} - \mathbf{x} \right)~,
```
where ``K(\mathbf{r})`` is a compact support kernel function, the sum runs over all ``N`` notional particles (indexed by ``k``), ``Q^{\ast (k)}`` is any particle property, ``\mathcal{V}^{\ast (k)}`` is the lumped volume represented by the particle, ``\mathbf{x}^{\ast (k)}`` is the position of particle ``k``, and ``\mathbf{x}`` and ``t`` are the location and time at which the mean variable is estimated.

In order for the mean estimation to be unbiased, the kernel function must hold some special properties: it must peak at ``\mathbf{r} = 0``, be normalised to unity, and have continuity properties. Because the volume ``\mathcal{V}^{\ast (k)}`` appears in the sum, the scheme is naturally more adapted for density-averaged means. If ``\Delta m`` is the mass of the particles (assumed constant, even if it is not mandatory), we have simply
```math
\widetilde{Q}(\mathbf{x}, t) \equiv \dfrac{\overline{\rho Q}}{\overline{\rho}} = \sum_{k=1}^N \Delta m Q^{\ast (k)} K \left( \mathbf{x}^{\ast (k)} - \mathbf{x} \right)~,
```
where the mean density is
```math
\overline{\rho}(\mathbf{x}, t) = \sum_{k=1}^N \Delta m K \left( \mathbf{x}^{\ast (k)} - \mathbf{x} \right)~.
```

Spatial derivatives of ``\overline{Q}`` can be estimated by the same token. For example:
```math
\bm{\nabla}\overline{Q}(\mathbf{x}, t) = -\sum_{k=1}^N \mathcal{V}^{\ast (k)} Q^{\ast (k)} \bm{\nabla} K \left( \mathbf{x}^{\ast (k)} - \mathbf{x} \right)~.
```

## Efficient algorithm

Solving the flow equations requires mean flow quantities to be estimated at each particle location, which can become computationally expensive, because the brute-force algorithm has complexity ``O\left( N^2 \right)``. The code implements instead the ``O(N)`` algorithm of [^Welton1998], which relies on a spatial Fourier decomposition of the kernel function. Note that it only works if it is separable, in the sense that, for a 2D domain
```math
K(\mathbf{r}) = K(r_x, h_x) \times K(r_y, h_y)~,
```
where ``r_x`` and ``r_y`` are the two components of the 2D distance vector, and ``h_x`` and ``h_y`` are the size of the kernel function in the two directions. In general, the kernel function takes a universal form when written in terms of ``r_x / h_x`` or ``r_y / h_y``, and only ``h_x`` and ``h_y`` must be adapted to the typical length scale of the flow. The limit of ``K`` when ``h`` tends to 0 must be the Dirac distribution.

The ``O(N)`` algorithm requires the 2D domain to be subdivided into rectangular subregions, which we refer to as R cells in the following, of size ``h_x \times h_y``. Means estimated at any location within a R cell only involve particles located in a larger, ``3h_x \times 3h_y`` subregion of the domain, which we refer to as S cells in the following. R cells are meant to pave the domain, while S cells, being comprised of 9 adjacent R cells, overlap each other.

At the root of the ``O(N)`` algorithm lies the fact that the argument passed to the kernel function is bounded, so that ``K`` can be expanded as a Fourier series. If ``K`` is symmetric, the periodicity of the series can be set to ``l_x = 3h_x`` (resp. ``l_y = 3h_y``), and we may write
```math
K(r_x, h_x) = \sum_{p=0}^M a_p \cos\left( \dfrac{2\pi p r_x}{l_x} \right)
```
(resp. for the ``y`` axis). Then, if the location ``\mathbf{x}`` at which the mean is to be estimated is in cell ``\mathcal{R}``, and we call ``\mathcal{S}`` the S cell that has ``\mathcal{R}`` as its central R cell, we may write
```math
\overline{\rho Q}(\mathbf{x}, t) = \sum_{p=0}^M \sum_{q=0}^M a_p a_q \left[ c_p c_q \left( \sum_{i \in \mathcal{S}} \Delta m Q^{\ast (i)} c_p^i c_q^i \right) + c_p s_q \left( \sum_{i \in \mathcal{S}} \Delta m Q^{\ast (i)} c_p^i s_q^i \right) + s_p c_q \left( \sum_{i \in \mathcal{S}} \Delta m Q^{\ast (i)} s_p^i c_q^i \right) + s_p s_q \left( \sum_{i \in \mathcal{S}} \Delta m Q^{\ast (i)} s_p^i s_q^i \right) \right]~,
```
where
```math
c_p = \cos\left( \dfrac{2\pi p x}{l_x} \right)~, s_p = \sin\left( \dfrac{2\pi p x}{l_x} \right)~, \\
c_q = \cos\left( \dfrac{2\pi q y}{l_y} \right)~, s_q = \sin\left( \dfrac{2\pi q y}{l_y} \right)~,
```
and ``c_p^i``, ``s_p^i``, ``c_q^i`` and ``s_q^i`` are the equivalent with the 2D coordinates of particle ``i`` instead.

The algorithm therefore runs thus: for each variable ``Q^{\ast}``
* for each vertical mode ``p`` and horizontal mode ``q``, the following sums, which we call *R sums* in the following, are computed
  ```math
  \sum_{i \in \mathcal{R}} \Delta m Q^{\ast (i)} c_p^i c_q^i~, \sum_{i \in \mathcal{R}} \Delta m Q^{\ast (i)} c_p^i s_q^i~, \sum_{i \in \mathcal{R}} \Delta m Q^{\ast (i)} s_p^i c_q^i~, \sum_{i \in \mathcal{R}} \Delta m Q^{\ast (i)} s_p^i s_q^i~.
  ```
  This is done for each R cell separately, and required the ``N`` particles to be binned according to which R cell they belong to.
* using the *R sums* computed above, we compute the following *S sums*
  ```math
  \sum_{i \in \mathcal{S}} \Delta m Q^{\ast (i)} c_p^i c_q^i~, \sum_{i \in \mathcal{S}} \Delta m Q^{\ast (i)} c_p^i s_q^i~, \sum_{i \in \mathcal{S}} \Delta m Q^{\ast (i)} s_p^i c_q^i~, \sum_{i \in \mathcal{S}} \Delta m Q^{\ast (i)} s_p^i s_q^i~,
  ```
  which simply involves summing the *R sums* over all R cells inside the S cell.
* for each location where ``\overline{Q}`` must be estimated, we find the appropriate S cell, and we apply the sum over ``p`` and ``q`` described above, where the *S sums* have already been computed in the previous step.

Both the first and third step have complexity ``O(N)``, and the second step is negligible. In fact, to be more precise, the whole algorithm has complexity ``O\left( N M^2 \right)``, because the different modes contribution must be added together. However, in practice, a maximum mode ``M = 10`` is largely enough to accurately represent the kernel function.

[^Welton1998]: Welton, W. C. 1998, Journal of Computational Physics, 139, 410