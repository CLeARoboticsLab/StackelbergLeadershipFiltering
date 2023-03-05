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

function add_control_cost!(c::QuadraticCost, other_player_idx, R; r=zeros(size(R, 1))::AbstractVector{Float64}, cr=0.::Float64)
    @assert size(R, 1) == size(R, 2) == size(r, 1)
    @assert size(cr) == ()

    c.Rs[other_player_idx] = R
    c.rs[other_player_idx] = r
    c.crs[other_player_idx] = cr
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


# Helpers that get the homogenized Q and R matrices for this cost.
function get_homogenized_state_cost_matrix(c::QuadraticCost)
    return homogenize_cost_matrix(c.Q, c.q, c.cq)
end

function get_homogenized_control_cost_matrix(c::QuadraticCost, player_idx::Int)
    return homogenize_cost_matrix(c.Rs[player_idx], c.rs[player_idx], c.crs[player_idx])
end

export get_homogenized_state_cost_matrix, get_homogenized_control_cost_matrix


# Derivative terms
function dgdx(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return x' * c.Q + c.q'
end

function dgdu(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return [us[ii]' * R + c.rs[ii] for (ii, R) in c.Rs]
end

function d2gdx2(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return c.Q
end

function d2gdu2(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return [R for (ii, R) in c.Rs]
end

export dgdx, dgdu, d2gdx2, d2gdu2


# Helpers specific to quadratic costs.
function get_quadratic_state_cost_term(c::QuadraticCost)
    return c.Q
end

function get_linear_state_cost_term(c::QuadraticCost)
    return c.q
end

function get_constant_state_cost_term(c::QuadraticCost)
    return c.cq
end

function get_quadratic_control_cost_term(c::QuadraticCost, player_idx::Int)
    return c.Rs[player_idx]
end

function get_linear_control_cost_term(c::QuadraticCost, player_idx::Int)
    return c.rs[player_idx]
end

function get_constant_control_cost_term(c::QuadraticCost, player_idx::Int)
    return c.crs[player_idx]
end

# Export all the cost type.
export QuadraticCost

# Export the helpers.
export get_quadratic_state_cost_term, get_linear_state_cost_term, get_constant_state_cost_term,
       get_quadratic_control_cost_term, get_linear_control_cost_term, get_constant_control_cost_term

# Export all the cost types/structs and functionality.
export add_control_cost!, quadraticize_costs, compute_cost
