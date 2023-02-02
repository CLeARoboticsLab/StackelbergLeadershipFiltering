# Utilities for managing quadratic and nonquadratic dynamics.
abstract type Cost end

# Every Cost is assumed to have the following functions defined on it:
# - affinize_costs(cost, time_range, x, us) - this function produces a QuadraticCost at time t given the state and controls
# - compute_cost(cost, xs, us) - this function evaluates the cost of a trajectory given the states and controls
# - homogenize_state(cost, xs) - needs to be defined if cost requires linear/constant terms
# - homogenize_ctrls(cost, us) - needs to be defined if cost requires linear/constant terms

# and has the following fields:
# - is_homogenized

# We use this type by making a substruct of it which can then have certain functions defined for it.
abstract type NonQuadraticCost <: Cost end

# Homogenize state - by default, this adds a 1 to the bottom. If a custom one is needed, define it elsewhere.
function homogenize_state(c::Cost, xs::AbstractMatrix{Float64})
    return vcat(xs, ones(1, size(xs, 2)))
end

function homogenize_ctrls(c::Cost, us::AbstractVector{<:AbstractMatrix{Float64}})
    num_players = length(us)
    return [vcat(us[ii], ones(1, size(us[ii], 2))) for ii in 1:num_players]
end


# Evaluates the cost across a time horizon.
# - xs[:, time]
# - us[player][:, time]
function evaluate(c::Cost, xs::AbstractMatrix{Float64}, us::AbstractVector{<:AbstractMatrix{Float64}})
    horizon = last(size(xs))

    # homogenize the output if needed
    if c.is_homogenized
        xs = homogenize_state(c, xs)
        us = homogenize_ctrls(c, us)
    end

    total = 0.0
    num_players = size(us, 1)
    for tt in 1:horizon
        prev_time = (tt == 1) ? tt : tt-1
        us_tt = [us[ii][:, tt] for ii in 1:num_players]
        total += compute_cost(c, (prev_time, tt), xs[:, tt], us_tt)
    end
    return total
end

# Export all the cost types/structs.
export Cost, NonQuadraticCost, evaluate
