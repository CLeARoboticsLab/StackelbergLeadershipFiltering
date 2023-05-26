# Affine costs with quadratic, linear, constant terms.

mutable struct QuadraticCost <: Cost
    Q::AbstractMatrix{Float64}
    q::AbstractVector{Float64}
    cq::Float64
    Rs::Dict{Int, Matrix{Float64}}
    rs::Dict{Int, Vector{Float64}}
    crs::Dict{Int, Float64}
end
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

# TODO(hmzh) - make pure quadratic code more detailed
function is_pure_quadratic(c::QuadraticCost)
    is_state_cost_pure_quadratic = all(iszero.(c.q) && iszero(c.cq))
    is_control_cost_pure_quadratic = all([all(iszero.(c.r[ii]) && iszero(c.cr[ii])) for (ii, r) in c.rs])
    return is_state_cost_pure_quadratic && is_control_cost_pure_quadratic
end

function compute_cost(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(us)
    total = (1/2.) * (x' * c.Q * x + 2 * c.q' * x + c.cq)
    if !isempty(c.Rs)
        total += (1/2.) * sum(us[jj]' * R * us[jj] + 2 * us[jj]' * c.rs[jj] + c.crs[jj] for (jj, R) in c.Rs)
    end
    return total
end

export is_pure_quadratic


# Derivative terms
function Gx(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return x' * c.Q + c.q'
end

function Gus(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return Dict(ii => us[ii]' * R + c.rs[ii]' for (ii, R) in c.Rs)
end

function Gxx(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return deepcopy(c.Q)
end

function Guus(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return deepcopy(c.Rs)
end

export Gx, Gus, Gxx, Guus


# Helpers specific to quadratic costs.
function get_quadratic_state_cost_term(c::QuadraticCost)
    return deepcopy(c.Q)
end

function get_linear_state_cost_term(c::QuadraticCost)
    return deepcopy(c.q)
end

function get_constant_state_cost_term(c::QuadraticCost)
    return deepcopy(c.cq)
end

function get_quadratic_control_cost_term(c::QuadraticCost, player_idx::Int)
    return deepcopy(c.Rs[player_idx])
end

function get_linear_control_cost_term(c::QuadraticCost, player_idx::Int)
    return deepcopy(c.rs[player_idx])
end

function get_constant_control_cost_term(c::QuadraticCost, player_idx::Int)
    return deepcopy(c.crs[player_idx])
end

# Export all the cost type.
export QuadraticCost

# Export the helpers.
export get_quadratic_state_cost_term, get_linear_state_cost_term, get_constant_state_cost_term,
       get_quadratic_control_cost_term, get_linear_control_cost_term, get_constant_control_cost_term

# Export all the cost types/structs and functionality.
export add_control_cost!, quadraticize_costs, compute_cost
