# Utilities for managing quadratic and nonquadratic dynamics.
abstract type Cost end

# Every Cost is assumed to have the following functions defined on it:
# - quadraticize_costs(cost, time_range, x, us) - this function produces a PureQuadraticCost at time t given the state and controls
# - compute_cost(cost, xs, us) - this function evaluates the cost of a trajectory given the states and controls
# - homogenize_state(cost, xs) - needs to be defined if cost requires linear/constant terms
# - homogenize_ctrls(cost, us) - needs to be defined if cost requires linear/constant terms

# No costs should require homogenized inputs. The evaluate function will homogenize the inputs as needed.

# We use this type by making a substruct of it which can then have certain functions defined for it.
abstract type NonQuadraticCost <: Cost end

# Produces a symmetric matrix.
# If we need to perform a spectral shift to enforce PD-ness, we can set ρ accordingly.
function homogenize_cost_matrix(M::AbstractMatrix{Float64}, m=zeros(size(M, 1))::AbstractVector{Float64}, cm=0.0::Float64, ρ=nothing)
    # If we're gonna have problems with singularity, then spectral shift the matrix.
    if all(m .== iszero.(size(M, 1))) && cm == 0.0 && ρ == nothing
        ρ = 1e-32
    elseif (ρ == nothing)
        ρ = 0.0
    end

    M_dim = size(M, 1)
    return vcat(hcat(M ,  m),
                hcat(m', cm)) + ρ * I
end

# Homogenize state - by default, this adds a 1 to the bottom. If a custom one is needed, define it elsewhere.
function homogenize_state(c::Cost, xs::AbstractArray{Float64})
    xhs = homogenize_vector(xs)
    return xhs
end

function homogenize_ctrls(c::Cost, us::AbstractVector{<:AbstractArray{Float64}})
    num_players = length(us)
    uhs = [homogenize_vector(us[ii]) for ii in 1:num_players]
    return uhs
end
export homogenize_state, homogenize_ctrls


# Evaluates the cost across a time horizon.
# - xs[:, time]
# - us[player][:, time]
function evaluate(c::Cost, xs::AbstractMatrix{Float64}, us::AbstractVector{<:AbstractMatrix{Float64}})
    horizon = last(size(xs))
    num_players = size(us, 1)

    # Homogenize the inputs.
    xs = homogenize_state(c, xs)
    us = homogenize_ctrls(c, us)

    total = 0.0
    for tt in 1:horizon
        prev_time = (tt == 1) ? tt : tt-1
        us_tt = [us[ii][:, tt] for ii in 1:num_players]
        total += compute_cost(c, (prev_time, tt), xs[:, tt], us_tt)
    end
    return total
end

# Export all the cost types/structs.
export Cost, NonQuadraticCost, evaluate
