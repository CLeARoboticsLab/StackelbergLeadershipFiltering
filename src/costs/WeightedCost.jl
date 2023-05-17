# A cost formed from by the weighted of other costs.
struct WeightedCost <: Cost
    weights::AbstractVector{<:Float64}
    costs::AbstractVector{<:Cost}
end

function compute_cost(c::WeightedCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(c.weights) == length(c.costs)
    return sum(c.weights[ii] * compute_cost(c.costs[ii], time_range, x, us) for ii in 1:length(c.weights))
end

# Derivative terms
function Gx(c::WeightedCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return sum(c.weights[ii] * Gx(c.costs[ii], time_range, x, us) for ii in 1:length(c.weights))
end

function Gus(c::WeightedCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(c.weights)
    Gs = [Gus(c_i, time_range, x, us) for c_i in c.costs]
    return Dict(jj => sum(c.weights[ii] * Gs[ii][jj] for ii in 1:length(c.weights)) for jj in 1:num_players)
end

function Gxx(c::WeightedCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return sum(c.weights[ii] * Gxx(c.costs[ii], time_range, x, us) for ii in 1:length(c.weights))
end

function Guus(c::WeightedCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(c.weights)
    Gs = [Guus(c_i, time_range, x, us) for c_i in c.costs]
    return Dict(jj => sum(c.weights[ii] * Gs[ii][jj] for ii in 1:length(c.weights)) for jj in 1:num_players)
end

# TODO(hamzah) - fix the bug in quadraticization and remove this after that
function quadraticize_costs(c::WeightedCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = size(us, 1)
    quad_costs = [quadraticize_costs(c.costs[ii], time_range, x, us) for ii in 1:length(c.weights)]

    sum_Q = sum(c.weights[ii] * get_quadratic_state_cost_term(quad_costs[ii]) for ii in 1:length(c.weights))
    sum_q = sum(c.weights[ii] * get_linear_state_cost_term(quad_costs[ii]) for ii in 1:length(c.weights))
    sum_cq = sum(c.weights[ii] * get_constant_state_cost_term(quad_costs[ii]) for ii in 1:length(c.weights))

    q_cost = QuadraticCost(sum_Q, sum_q, sum_cq)

    for jj in 1:num_players
        sum_R = sum(c.weights[ii] * get_quadratic_control_cost_term(quad_costs[ii], jj) for ii in 1:length(c.weights))
        sum_r = sum(c.weights[ii] * get_linear_control_cost_term(quad_costs[ii], jj) for ii in 1:length(c.weights))
        sum_cr = sum(c.weights[ii] * get_constant_control_cost_term(quad_costs[ii], jj) for ii in 1:length(c.weights))

        add_control_cost!(q_cost, jj, sum_R; r=sum_r, cr=sum_cr)
    end

    return q_cost
end

# Export the derivatives.
export Gx, Gus, Gxx, Guus

# Export all the cost type.
export WeightedCost

# Export all the cost types/structs and functionality.
export compute_cost
