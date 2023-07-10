# coredot
Smart dot-product optimizer

This package provides two central pieces of logic for econometric
projections:

 - Multiple optimized dot-product calculators, which are
   differentiated by CPU/GPU architecture, calculation size, memory
   availability, and known duplication.
   
 - A parallization framework, which selects the best optimized
   dot-product calculator given initialization-time features like
   those above and provides a general framework for handling
   calculations based on what can and cannot be parallized.
   
## Context

The "inner loop" of our econometric projections that can be most
thoroughly optimized is the dot product operation across all regions
and possibly other dimensions. In some cases, this can be parallelized
with the GPU, although that decision can be made at run-time based on
the properties of the data.

The parallel dot-product operation will be compartmentalized to
maximize the opportunities for parallel, vectorized, and compiled
optimizations.

At initialization, the features of the dot product operation will be
set, including:

- Dimension and sizes (e.g., number of weather variables; see parallelization above)
- Known duplication (e.g., weather or coefficients used across multiple regions)
- Input and output memory contexts (e.g., CPU or GPU)

This allows the core to develop an execution plan. We will estimate
dask, numba, and GPU approaches to understand when each is
appropriate.

Pointers to the relevant weather will generally be provided for all
regions at once, and possibly left in as multi-file xarrays or copied
to the GPU depending on the execution plan.

The memory context is important because memory transfer into and out
of the GPU makes simple dot products inefficient on GPUs, relative to
CPUs
(https://blog.theincredibleholk.org/blog/2012/12/10/optimizing-dot-product/). However,
if the data can be used for multiple dot products, we can get
benefits. This would occur if we do grid-level calculations, where
coefficients are duplicated and where regional aggregation is also
needed and can be done on the GPU. We can also find opportunities with
larger arrays (https://dournac.org/info/gpu_sum_reduction).

## Dot-product selection

We will handle the following cases:
 - A single set of coefficients is applied to many observations (e.g.,
   Labor).
 - Each observation has a different set of coefficients (e.g.,
   Mortality).
 - Some coefficients are shared while others are observation-specific
   (e.g., Agriculture). More generally, a final calculation is a sum
   of sets of coefficients from the first two cases.
   
The third case makes things interesting, because we need to know if
the grouping follows a common pattern across sets of coefficients. For
example, if one group of coefficients varies by Region and another by
Time, then each Region-Time observation will end up with a different
final set of coefficients. Or there could be two different groups of
coefficients, each of which varies by Region, but which do so
differently.

The information will be provided as two dictionaries:
 - `{group: #predictors}`: Specifies the groups of constant
   predictors. E.g., one could have {'USA': 3, 'IND': 3, ...} if the
   coefficients vary by region only, or {'USA-2020': 3, ...} if they
   vary by region-year. A special '*' group means that predictors vary
   by each observation.
 - `{(group1, group2, ...): #observations}`: Specifies the number of
   observations that get each sum of groups as their dot-product
   calculation.
