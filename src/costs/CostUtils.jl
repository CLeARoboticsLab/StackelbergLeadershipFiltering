# Utilities for managing quadratic and nonquadratic dynamics.
abstract type Cost end

# Every Cost is assumed to have the following functions defined on it:
# - quadraticize_costs(cost, time_range, x, us) - this function produces a QuadraticCost at time t given the state and controls
# - evaluate(cost, xs, us) - this function evaluates the cost of a trajectory given the states and controls

# We use this type by making a substruct of it which can then have certain functions defined for it.
abstract type NonQuadraticCost <: Cost end

# Export all the cost types/structs.
export Cost, NonQuadraticCost
