# MultiTankMaterialBalance

[![Build Status](https://github.com/sidelkin1/MultiTankMaterialBalance.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sidelkin1/MultiTankMaterialBalance.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/sidelkin1/MultiTankMaterialBalance.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sidelkin1/MultiTankMaterialBalance.jl)
[![DOI:10.1088/1755-1315/808/1/012034](https://img.shields.io/badge/DOI-10.1088%2F1755--1315%2F808%2F1%2F012034-blue)](https://doi.org/10.1088/1755-1315/808/1/012034)

*Read this in other languages: [English](README.md), [Русский](README.ru.md)*

The package implements a multi-tank material balance model. The model is designed to predict reservoir pressure in tanks, taking into account:

- Liquid production and water injection through wells inside the tanks
- Fluid flows between neighboring tanks due to pressure difference
- Water inflow from the aquifer through tanks at the outer boundary of the reservoir

## Package features:

- Implicit numerical scheme for calculating reservoir pressure in tanks based on the Newton method
- Analytic calculation of the jacobian
- Gradient calculation with respect to tank parameters based on the adjoint equation method
- Implementation of the objective function with automatic adjustment of well productivity/injectivity

## Interactive example of package using

To understand the theory and practice of working with the package [training example](https://github.com/sidelkin1/multitank-matbal-tutorial) was created based on Jupyter Notebook.

## Command language interface (CLI)

A [command line interface](https://github.com/sidelkin1/multitank-matbal-cli) (CLI) is created for working with the package. It implements automatic fitting of tank parameters.

## Citation

If you use `MultiTankMaterialBalance` package in your research, please cite this paper:

```tex
@article{article,
author = {Sidelnikov, K and Tsepelev, V and Kolida, A and Faizullin, R},
year = {2021},
month = {07},
pages = {012034},
title = {Reservoir energy management based on the method of multi-tank material balance},
volume = {808},
journal = {IOP Conference Series: Earth and Environmental Science},
doi = {10.1088/1755-1315/808/1/012034}
}
```