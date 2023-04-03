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
    Gs = [Gus(c.costs[ii], time_range, x, us) for ii in 1:num_players]
    return Dict(jj => sum(c.weights[jj] * G[jj] for G in Gs) for jj in 1:num_players)
end

function Gxx(c::WeightedCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return sum(c.weights[ii] * Gxx(c.costs[ii], time_range, x, us) for ii in 1:length(c.weights))
end

function Guus(c::WeightedCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(c.weights)
    Gs = [Guus(c.costs[ii], time_range, x, us) for ii in 1:num_players]
    return Dict(jj => sum(c.weights[jj] * G[jj] for G in Gs) for jj in 1:num_players)
end

# Export the derivatives.
export Gx, Gus, Gxx, Guus

# Export all the cost type.
export WeightedCost

# Export all the cost types/structs and functionality.
export compute_cost
