# Utilities for managing quadratic and nonquadratic dynamics.
abstract type Cost end

# Every Cost is assumed to have the following functions defined on it:
# - quadraticize_costs(cost, time_range, x, us) - this function produces a PureQuadraticCost at time t given the state and controls
# - compute_cost(cost, xs, us) - this function evaluates the cost of a trajectory given the states and controls
# - Gx, Gus, Gxx, Guus(cost, time_range, x, us) - derivatives must be defined.

# No costs should require homogenized inputs. The evaluate function will homogenize the inputs as needed.

# We use this type by making a substruct of it which can then have certain functions defined for it.
abstract type NonQuadraticCost <: Cost end

# Produces a symmetric matrix.
# If we need to perform a spectral shift to enforce PD-ness, we can set ρ accordingly.
function homogenize_cost_matrix(M::AbstractMatrix{Float64}, m=zeros(size(M, 1))::AbstractVector{Float64}, cm=0.0::Float64, ρ=nothing)
    # If we're gonna have problems with singularity, then spectral shift the matrix.
    if all(iszero.(m)) && cm == 0.0 && ρ == nothing
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

    total = 0.0
    for tt in 1:horizon
        prev_time = (tt == 1) ? tt : tt-1
        us_tt = [us[ii][:, tt] for ii in 1:num_players]
        total += compute_cost(c, (prev_time, tt), xs[:, tt], us_tt)
    end
    return total
end

# Return the second order Taylor approximation for dx = x - x0:
# g(x0 + dx, u0s + dus) ≈ f(x, us) + dx' ∇ₓ g(x0, u0s) + (1/2) dx' ∇ₓₓ g(x0, u0s) dx
#                                  + ∑ du' ∇ᵤ g(x0, u0s) + (1/2) du' ∇ᵤᵤ g(x0, u0s) du
function quadraticize_costs(c::Cost, time_range, x0::AbstractVector{Float64}, u0s::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(us)

    cost_eval = compute_cost(c, time_range, x0, u0s)
    ddx2 = Gxx(c, time_range, x0, u0s)
    dx = Gx(c, time_range, x0, u0s)
    ddu2s = Guus(c, time_range, x0, u0s)
    dus = Gus(c, time_range, x0, u0s)

    # Used to compute the way the constant cost terms are divided.
    num_cost_mats = length(ddu2s)
    const_cost_term = (2/num_cost_mats) * cost_eval

    # This should be QuadraticCost with offset about x because the taylor approx is (x-x0)
    quad_cost = QuadraticCost(ddx2, dx', const_cost_term)
    for (ii, du) in dus
        add_control_cost!(quad_cost, ii, ddu2s[ii]; r=dus[ii]', cr=const_cost_term)
    end

    # offset_cost = QuadraticCostWithOffset(quad_cost, x0, u0s)

    return quad_cost
end


# Export all the cost types/structs.
export Cost, NonQuadraticCost, evaluate
