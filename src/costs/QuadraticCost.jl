# Affine costs with quadratic, linear, constant terms.

mutable struct QuadraticCost <: Cost
    Q::AbstractMatrix{Float64}
    q::AbstractVector{Float64}
    cq::Float64
    Rs::Dict{Int, Matrix{Float64}}
    rs::Dict{Int, Vector{Float64}}
    crs::Dict{Int, Float64}
end
# Quadratic costs always homogeneous
QuadraticCost(Q::AbstractMatrix{Float64}) = QuadraticCost(Q, zeros(size(Q, 1)), 0,
                                                          Dict{Int, Matrix{eltype(Q)}}(),
                                                          Dict{Int, Vector{eltype(Q)}}(),
                                                          Dict{Int, eltype(Q)}())
QuadraticCost(Q::AbstractMatrix{Float64}, q::AbstractVector{Float64}, cq::Float64) = QuadraticCost(Q, q, cq,
                                                                                                   Dict{Int, Matrix{eltype(Q)}}(),
                                                                                                   Dict{Int, Vector{eltype(q)}}(),
                                                                                                   Dict{Int, eltype(cq)}())

function add_control_cost!(c::QuadraticCost, other_player_idx, Rij; rj=zeros(size(Rij, 1))::AbstractVector{Float64}, crj=0.::Float64)
    @assert size(Rij, 1) == size(Rij, 2) == size(rj, 1)
    @assert size(crj) == ()

    c.Rs[other_player_idx] = Rij
    c.rs[other_player_idx] = rj
    c.crs[other_player_idx] = crj
end

function quadraticize_costs(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    Q̃ = homogenize_cost_matrix(c.Q, c.q, c.cq)
    q_cost = PureQuadraticCost(Q̃)

    # Fill control costs.
    num_players = length(c.Rs)
    for ii in 1:num_players
        R̃ = homogenize_cost_matrix(c.Rs[ii], c.rs[ii], c.crs[ii];)
        add_control_cost!(q_cost, ii, R̃)
    end

    return q_cost
end

function compute_cost(c::QuadraticCost, time_range, xh::AbstractVector{Float64}, uhs::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(uhs)
    out_size = size(c.Q, 1)
    x = xh[1:out_size]

    total = (1/2.) * x' * c.Q * x + c.q' * x + (1/2.) * c.cq
    for ii in 1:num_players
        out_size = size(c.Rs[ii], 1)
        u = uhs[ii][1:out_size]

        total += (1/2.) * u' * c.Rs[ii] * u + c.rs[ii]' * u + (1/2.) * c.crs[ii]
    end
    return total
end

# Export all the cost type.
export QuadraticCost

# Export functionality.
export add_control_cost!, quadraticize_costs, compute_cost
