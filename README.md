# Hawkes Plastic Networks

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dylanfesta.github.io/HawkesPlasticNetworks.jl/dev/)
[![Build Status](https://github.com/dylanfesta/HawkesPlasticNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dylanfesta/HawkesPlasticNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-white.svg)](LICENSE)

This package allows to build networks of Poisson units that can interact
by either potentiating (excitatory) or depressing (inhibitory) each other.
The kernel used is always an exponential kernel.

Although the dynamics are narrowly chosen, this package focuses on
high performance and on synaptic plasticity, offering a wide choice
of plasticity mechanisms that modify the connectivity matrix dynamically.

Some of the examples are taken from the paper below. Please cite it if you
appreciate this work ❤️

> Festa, D., Cusseddu, C. and Gjorgjieva, J. (2026) “Structured stabilization in recurrent neural circuits through inhibitory synaptic plasticity,” *eLife*, 15. Available at: [doi:10.7554/eLife.111666.1](https://doi.org/10.7554/eLife.111666).

!!! warning
  Work in progress! Feel free to open issues if you want more features of if you notice something wrong.
