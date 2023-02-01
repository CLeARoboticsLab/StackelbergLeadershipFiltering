# Affine costs with quadratic, linear, constant terms.

mutable struct AffineCost <: Cost
    Q::AbstractMatrix{Float64}
    q::AbstractVector{Float64}
    cq::Float64
    Rs::Dict{Int, Matrix{Float64}}
    rs::Dict{Int, Vector{Float64}}
    crs::Dict{Int, Float64}
end
AffineCost(Q, q, cq) = AffineCost(Q, q, cq, Dict{Int, Matrix{eltype(Q)}}(), Dict{Int, Vector{eltype(q)}}(), Dict{Int, eltype(cq)}())

function add_control_cost!(c::AffineCost, other_player_idx, Rij, rj, crj)
    @assert size(Rij, 1) == size(Rij, 2) == size(rj, 1)
    @assert size(crj) == ()

    c.Rs[other_player_idx] = Rij
    c.rs[other_player_idx] = rj
    c.crs[other_player_idx] = crj
end

function affinize_costs(cost::AffineCost, time_range, x, us)

    return cost

    # num_players = size(us, 1)
    # Q = homogenize_matrix(cost.Q, cost.q, cost.cq)
    # cost = QuadraticCost(Q)
    # for jj in 1:num_players
    #     R_ij = homogenize_matrix(cost.Rs[jj], cost.rs[jj], cost.crs[jj])
    #     add_control_cost!(cost, R_ij)
    # end
    # return cost
end

# TODO: Implement a way to do this well.
function evaluate(c::AffineCost, xs, us)
    error("Affine cost evaluation not implemented. Please use affinize_costs to extract a QuadraticCost and pad states with an extra entry of 1.")
end

# Export all the cost type.
export AffineCost

# Export functionality.
export affinize_costs, evaluate, homogenize_matrix
