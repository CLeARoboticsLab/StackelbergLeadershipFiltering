# Utilities for managing quadratic and nonquadratic dynamics.
abstract type Cost end

# Every Cost is assumed to have the following functions defined on it:
# - quadraticize_costs(cost, t, x, us) - this function produces a QuadraticCost at time t given the state and controls
# - evaluate(cost, xs, us) - this function evaluates the cost of a trajectory given the states and controls

# We use this type by making a substruct of it which can then have certain functions defined for it.
abstract type NonQuadraticCost <: Cost end

# Cost for a single player.
# Form is: x^T_t Q^i x + \sum_j u^{jT}_t R^{ij} u^j_t.
# For simplicity, assuming that Q, R are time-invariant, and that dynamics are
# linear time-invariant, i.e. x_{t+1} = A x_t + \sum_i B^i u^i_t.
mutable struct QuadraticCost <: Cost
    Q::AbstractMatrix{Float64}
    Rs
end
QuadraticCost(Q) = QuadraticCost(Q, Dict{Int, Matrix{eltype(Q)}}())

# TODO(hamzah) Add better tests for the QuadraticCost struct and associated functions.

# Method to add R^{ij}s to a Cost struct.
export add_control_cost!
function add_control_cost!(c::QuadraticCost, other_player_idx, Rij)
    c.Rs[other_player_idx] = Rij
end

function quadraticize_costs(cost::QuadraticCost, t, x, us)
    return cost
end

# Evaluate cost on a state/control trajectory.
# - xs[:, time]
# - us[player][:, time]
function evaluate(c::QuadraticCost, xs, us)
    horizon = last(size(xs))

    total = 0.0
    for tt in 1:horizon
        total += xs[:, tt]' * c.Q * xs[:, tt]
        total += sum(us[jj][:, tt]' * Rij * us[jj][:, tt] for (jj, Rij) in c.Rs)
    end
    return total
end


function homogenize_cost(M::AbstractMatrix{Float64}, m::AbstractVector{Float64}, cm::Float64)
    M_dim = size(M, 1)
    return vcat(hcat(M , zeros(M_dim, 1)),
                hcat(m',              cm))
end

mutable struct AffineCost <: Cost
    Q::AbstractMatrix{Float64}
    q::AbstractVector{Float64}
    cq::Float64
    Rs::Dict{Int, Matrix{eltype(Q)}}
    rs::Dict{Int, Vector{eltype(q)}}
    crs::Dict{Int, eltype(cq)}
end
AffineCost(Q, q, cq) = AffineCost(Q, q, cq, Dict{Int, Matrix{eltype(Q)}}(), Dict{Int, Vector{eltype(q)}}(), Dict{Int, eltype(cq)}())

function add_control_cost!(c::AffineCost, other_player_idx, Rij, rj, crj)
    @assert size(Rij, 1) == size(Rij, 2) == size(rj, 1)
    @assert size(crj) == ()

    c.Rs[other_player_idx] = Rij
    c.rs[other_player_idx] = rj
    c.crs[other_player_idx] = crj
end

function quadraticize_costs(cost::AffineCost, t, x, us)
    num_players = size(us, 1)
    Q = homogenize_cost(cost.Q, cost.q, cost.cq)
    cost = QuadraticCost(Q)
    for jj in 1:num_players
        R_ij = homogenize_cost(cost.Rs[jj], cost.rs[jj], cost.crs[jj])
        add_control_cost!(cost, R_ij)
    end
    return cost
end

# TODO: Implement a way to do this well.
function evaluate(c::AffineCost, xs, us)
    error("Affine cost evaluation not implemented. Please use quadraticize_costs to extract a QuadraticCost and pad states with an extra entry of 1.")
end

# Export all the cost types/structs.
export Cost, NonQuadraticCost, QuadraticCost, AffineCost

# Export all the cost types/structs.
export quadraticize_costs, evaluate, homogenize_cost

