# MS Report
This repo is intended to hold the data and the code required to reproduce the entire project.

Majority of the code uses python and jupyter notebooks (hosted on colab). Google colab provides an easy medium for code hosting, convenience and reproducibility. Data present in this repo is made available in individual notebook by cloning this repo appropriately.


# Title
Multiscale Modelling of CuAgAu surface segregation for propylene epoxidation

## Abstract

Multicomponent alloys very often exhibit superior properties than their constituent individual metals. For an industrially relevant reaction such as propylene epoxidation, an appropriate catalyst can greatly enhance the selectivity and conversion of the process. For example, Ag based catalysts show excellent activity but alloying Ag with Cu or Au exhibit increased selectivity and conversion. We postulate that there exists an alloy of composition CuAgAu which possess desired target property or some combination of those properties. In the pursuit of finding this optimum composition, one needs to be aware of effects of the phenomenon of surface segregation which states that the surface composition of a multicomponent alloy often differs from its bulk composition. This deviation depends on bulk concentrations, temperature, pressure and adsorbates present on surface. This study is focussed on understanding the extent of surface segregation on all possible compositions in CuAgAu ternary system. Fast computational methods allow us to accurately predict the changes due to surface segregation for any given bulk composition. In this work, surface segregation is modelled via Monte Carlo simulations in canonical ensemble. To evaluate energies of intermediate configurations arising during Monte Carlo simulations, a machine learning model is trained on first principles Density Functional Theory (DFT) energies. Combination of these computational tools lets us predict the deviation of surface composition from bulk composition of a multicomponent alloy.

## Overview of this repository
The project is structured as follows:

1. _k_ pts convergence tests
2. Planewave cutoff convergence tests
3. Bulk lattice constant calculations
4. Behler Parinello Neural Network training
5. Monte Carlo simulations
6. Visualizing surface excess results
7. Investigating AgAu segregation variation with van der waals forces correction
8. Plotting histogram for near-surface atomic swaps with energy of swap
9. Validating neural network performance on surface relaxed structures
10. Evaluating surface segregation trend for FCC211 surface
