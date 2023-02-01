# Utilities for managing quadratic and nonquadratic dynamics.
abstract type Cost end

# Every Cost is assumed to have the following functions defined on it:
# - affinize_costs(cost, time_range, x, us) - this function produces a QuadraticCost at time t given the state and controls
# - evaluate(cost, xs, us) - this function evaluates the cost of a trajectory given the states and controls
# - homogenize_state(cost, xs) - needs to be defined if cost requires linear/constant terms
# - homogenize_ctrls(cost, us) - needs to be defined if cost requires linear/constant terms

# and has the following fields:
# - is_homogenized

# We use this type by making a substruct of it which can then have certain functions defined for it.
abstract type NonQuadraticCost <: Cost end

# Export all the cost types/structs.
export Cost, NonQuadraticCost
