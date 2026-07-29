# Hawkes Plastic Networks

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dylanfesta.github.io/HawkesPlasticNetworks.jl/dev/)
[![Build Status](https://github.com/dylanfesta/HawkesPlasticNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dylanfesta/HawkesPlasticNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)

This package allows to build networks of Poisson units that can interact
by either potentiating (excitatory) or depressing (inhibitory) each other. 
The kernel used is always an exponential kernel.

Although the dynamics are narrowly chosen, this package focuses on 
high performance and on synaptic plasticity, offering a wide choice
of plasticity mechanisms that modify the connectivity matrix dynamically.


!!! warning
  This is a simplified and optimized version of HawkesSimulator.jl. Several elements are still missing.

Work in progress!
