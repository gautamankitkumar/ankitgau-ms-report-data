[![HitCount](http://hits.dwyl.com/gautamankitkumar/ankitgau-ms-report-data.svg)](http://hits.dwyl.com/gautamankitkumar/ankitgau-ms-report-data) [![PyPi Python Versions](https://img.shields.io/pypi/pyversions/yt2mp3.svg)](https://www.python.org/) [![Twitter Follow](https://img.shields.io/twitter/follow/Gautam_K_Ankit.svg?style=social)](https://twitter.com/Gautam_K_Ankit)

# About
This repo is intended to hold the data and the code required to reproduce the entire project.

Majority of the code uses python and jupyter notebooks (hosted on colab). Google colab provides an easy medium for code hosting, convenience and reproducibility.

You can find the PDF version of the report [here](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/Gautam_AnkitKumar_F20.pdf)

# Title
Multiscale Modelling of CuAgAu surface segregation for propylene epoxidation

# Graphical Abstract


![Multiscale Modelling](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/data/graphical-abstract.png)

Credits to [![Twitter Follow](https://img.shields.io/twitter/follow/RailkarRucha.svg?style=social)](https://twitter.com/RailkarRucha)

# Abstract

Multicomponent alloys very often exhibit superior properties than their constituent individual metals. For an industrially relevant reaction such as propylene epoxidation, an appropriate catalyst can greatly enhance the selectivity and conversion of the process. For example, Ag based catalysts show excellent activity but alloying Ag with Cu or Au exhibit increased selectivity and conversion. We postulate that there exists an alloy of composition CuAgAu which possess desired target property or some combination of those properties. In the pursuit of finding this optimum composition, one needs to be aware of effects of the phenomenon of surface segregation which states that the surface composition of a multicomponent alloy often differs from its bulk composition. This deviation depends on bulk concentrations, temperature, pressure and adsorbates present on surface. This study is focussed on understanding the extent of surface segregation on all possible compositions in CuAgAu ternary system. Fast computational methods allow us to accurately predict the changes due to surface segregation for any given bulk composition. In this work, surface segregation is modelled via Monte Carlo simulations in canonical ensemble. To evaluate energies of intermediate configurations arising during Monte Carlo simulations, a machine learning model is trained on first principles Density Functional Theory (DFT) energies. Combination of these computational tools lets us predict the deviation of surface composition from bulk composition of a multicomponent alloy.

# Overview of this repository
The project is structured as follows:

- [Bulk lattice constant calculation](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/notebooks/bulk-lattice.ipynb)
- [DFT training data generation](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/notebooks/generate-DFT-configs.ipynb)
- [Behler-Parinello Neural Network training](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/notebooks/train-BPNN.ipynb)
- [Monte Carlo simulations](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/notebooks/run-mc-simulation.ipynb)
- [Visualizing surface excess results](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/notebooks/surface-excess.ipynb)
- Investigating AgAu segregation variation with van der waals forces correction
- [Plotting histogram for near-surface atomic swaps with energy of swap](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/notebooks/swap-histogram.ipynb)
- [Validating neural network performance on surface relaxed structures](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/notebooks/surface-relax.ipynb)
- Evaluating surface segregation trend for FCC211 surface
- [Analytical Solution](https://github.com/gautamankitkumar/ankitgau-ms-report-data/blob/main/notebooks/analytical-solution.ipynb)


## TODO
- Include kpts convergence
- Verify all notebooks contain an executed output