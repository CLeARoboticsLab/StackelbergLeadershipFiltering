# Stackelberg Control Hypotheses Filtering

## Setup

This repo is structured as a Julia package. To activate this package, type
```console
julia> ]
(@v1.6) pkg> activate .
  Activating environment at `<path to repo>/Project.toml`
(StackelbergControlHypothesesFiltering) pkg>
```
Now exit package mode by hitting the `[delete]` key. You should see the regular Julia REPL prompt. Type:
```console
julia> using Revise
julia> using StackelbergControlHypothesesFiltering
```

## Utilities
Contains useful utilities. The `Cost` struct stores matrices which define a quadratic cost function for each player, and the `Dynamics` struct defines linear game dynamics (both are time-invariant). Functions to evaluate cost functions and unroll trajectories from initial conditions are provided as well.



To run tests locally and avoid polluting your commit history, in the REPL you can type:
```console
julia> ]
(StackelbergControlHypothesesFiltering) pkg> test
```

Alternatively, you can run:
```console
julia> include("test/runtests.jl")
```
