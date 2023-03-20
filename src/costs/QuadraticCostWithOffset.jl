# Affine costs with quadratic, linear, constant terms.

struct QuadraticCostWithOffset <: Cost
    q_cost::QuadraticCost
    x_dest::AbstractVector{Float64}
end
# Quadratic costs always homogeneous
QuadraticCostWithOffset(quad_cost::QuadraticCost, x_dest=zeros(size(quad_cost.Q, 1))) = QuadraticCostWithOffset(quad_cost, x_dest)

# TODO(hmzh): Adjust quadraticization for the multi-player case.
function quadraticize_costs(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    # @assert length(us) == 1
    Q = get_quadratic_state_cost_term(c.q_cost)
    q = Q * (x - c.x_dest)
    cq = c.x_dest' * Q * c.x_dest

    cost = QuadraticCost(Q, q, cq)
    for (ii, R) in c.q_cost.Rs
        add_control_cost!(cost, ii, c.q_cost.Rs[ii]; r=c.q_cost.Rs[ii] * us[ii] + c.q_cost.rs[ii], cr=c.x_dest' * Q * c.x_dest)
    end

    return cost
end

function compute_cost(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(us)
    dx = x - c.x_dest
    return compute_cost(c.q_cost, time_range, dx, us)
end

# Define derivative terms.
function Gx(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    Q = get_quadratic_state_cost_term(c.q_cost)
    return (x - c.x_dest)' * Q
end

function Gus(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return Dict(ii => us[ii]' * R + c.q_cost.rs[ii]' for (ii, R) in c.q_cost.Rs)
end

function Gxx(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return get_quadratic_state_cost_term(c.q_cost)
end

function Guus(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return deepcopy(c.q_cost.Rs)
end


# Export all the cost type.
export QuadraticCostWithOffset

# Export all the cost types/structs and functionality.
export add_control_cost!, quadraticize_costs, compute_cost

# Export derivative terms
export Gx, Gus, Gxx, Guus
