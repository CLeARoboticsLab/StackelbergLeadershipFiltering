# Affine costs with quadratic, linear, constant terms.

mutable struct AffineCost <: Cost
    Q::AbstractMatrix{Float64}
    q::AbstractVector{Float64}
    cq::Float64
    Rs::Dict{Int, Matrix{Float64}}
    rs::Dict{Int, Vector{Float64}}
    crs::Dict{Int, Float64}
    is_homogenized::Bool
end
# Affine costs always homogeneous
AffineCost(Q, q, cq) = AffineCost(Q, q, cq, Dict{Int, Matrix{eltype(Q)}}(), Dict{Int, Vector{eltype(q)}}(), Dict{Int, eltype(cq)}(), true)

function add_control_cost!(c::AffineCost, other_player_idx, Rij, rj, crj)
    @assert size(Rij, 1) == size(Rij, 2) == size(rj, 1)
    @assert size(crj) == ()

    c.Rs[other_player_idx] = Rij
    c.rs[other_player_idx] = rj
    c.crs[other_player_idx] = crj
end

function affinize_costs(cost::AffineCost, time_range, x, us)
    return cost
end

# TODO: Implement a way to do this well.
function compute_cost(c::AffineCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(us)
    Q̃ = homogenize_matrix(c.Q, c.q, c.cq)
    R̃s = [homogenize_matrix(c.Rs[ii], c.rs[ii], c.crs[ii]) for ii in 1:num_players]
    return x' * Q̃ * x + sum(us[jj]' * Rij * us[jj] for (jj, Rij) in R̃s)
end

# Export all the cost type.
export AffineCost

# Export functionality.
export add_control_cost!, affinize_costs, compute_cost
