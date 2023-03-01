# Affine costs with quadratic, linear, constant terms.

mutable struct QuadraticCost <: Cost
    Q::AbstractMatrix{Float64}
    q::AbstractVector{Float64}
    cq::Float64
    offset
    # TODO(hamzah): add offset for control costs when needed
    Rs::Dict{Int, Matrix{Float64}}
    rs::Dict{Int, Vector{Float64}}
    crs::Dict{Int, Float64}
end
# Quadratic costs always homogeneous
QuadraticCost(Q::AbstractMatrix{Float64}) = QuadraticCost(Q, zeros(size(Q, 1)), 0, nothing,
                                                          Dict{Int, Matrix{eltype(Q)}}(),
                                                          Dict{Int, Vector{eltype(Q)}}(),
                                                          Dict{Int, eltype(Q)}())
QuadraticCost(Q::AbstractMatrix{Float64}, q::AbstractVector{Float64}, cq::Float64) = QuadraticCost(Q, q, cq, nothing,
                                                                                                   Dict{Int, Matrix{eltype(Q)}}(),
                                                                                                   Dict{Int, Vector{eltype(q)}}(),
                                                                                                   Dict{Int, eltype(cq)}())
# Quadratic Cost with offset (i.e. quadratic error cost)
function make_quadratic_cost_with_offset(Q::AbstractMatrix{Float64},
                                         offset::AbstractVector{Float64})
    xf = offset
    return QuadraticCost(Q,
                         -Q*xf,
                         0.5*xf'*Q*xf,
                         xf,
                         Dict{Int, Matrix{eltype(Q)}}(),
                         Dict{Int, Vector{eltype(Q)}}(),
                         Dict{Int, eltype(Q)}())
end

function add_control_cost!(c::QuadraticCost, other_player_idx, R; r=zeros(size(R, 1))::AbstractVector{Float64}, cr=0.::Float64)
    @assert size(R, 1) == size(R, 2) == size(r, 1)
    @assert size(cr) == ()

    c.Rs[other_player_idx] = R
    c.rs[other_player_idx] = r
    c.crs[other_player_idx] = cr
end

function quadraticize_costs(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    new_q = (c.offset == nothing) ? c.q : c.Q * (x - c.offset)

    Q̃ = homogenize_cost_matrix(c.Q, new_q, c.cq)
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
    ctrl_sizes = [size(c.rs[ii], 1) for ii in 1:num_players]
    us = [uhs[ii][1:ctrl_sizes[ii]] for ii in 1:num_players]

    total = (1/2.) * (x' * c.Q * x + 2 * c.q' * x + c.cq)
    if !isempty(c.Rs)
        total += (1/2.) * sum(us[jj]' * R * us[jj] + 2 * us[jj]' * c.rs[jj] + c.crs[jj] for (jj, R) in c.Rs)
    end
    return total
end

# Export all the cost type.
export QuadraticCost, make_quadratic_cost_with_offset

# Export all the cost types/structs and functionality.
export add_control_cost!, quadraticize_costs, compute_cost, compute_control_cost
