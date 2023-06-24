# A cost f(x) = -log(x_i - a) which ensures that x_i > a. Or if a flag is set, that x_i < a.
# This is a 1D barrier on state values, independent of other dimensions.
struct AbsoluteLogBarrierCost <: NonQuadraticCost
    idx::Int             # a single index of the state
    offset::Real              # offset with size identical to x
    is_lower_bound::Bool # true indicates lower bound barrier, false indicates upper bound
end

const LARGE_NUMBER = 1e6

# Compute input to log function based on whether we have an upper or lower bound.
function _get_input(c::AbsoluteLogBarrierCost, x)
    return (c.is_lower_bound) ? x[c.idx] - c.offset : c.offset - x[c.idx]
end

function _violates_bound(c::AbsoluteLogBarrierCost, x)
    violates_lower_bound = c.is_lower_bound && x[c.idx] ≤ c.offset
    violates_upper_bound = !c.is_lower_bound && x[c.idx] ≥ c.offset
    return violates_upper_bound || violates_lower_bound
end

function compute_cost(c::AbsoluteLogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    if _violates_bound(c, x)
        return LARGE_NUMBER
    else
        input = _get_input(c, x)
        return -log(input)
    end
end

# Derivative term c(x_i - a_i)^-1 at {c.idx}th index.
function Gx(c::AbsoluteLogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    out = zeros(1, size(x, 1))

    # If we are in the wrong place, choose a gradient of 0 and hope constraints pull it back out.
    if _violates_bound(c, x)
        return out
    end

    out[1, c.idx] = -(x[c.idx] - c.offset)^(-1)
    return out
end

function Gus(c::AbsoluteLogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(us)
    return Dict(jj => zeros(1, size(us[jj], 1)) for jj in 1:num_players)
end

# diagonal matrix, with diagonals either 0 or c(x_i - a_i)^-2, no cross terms
function Gxx(c::AbsoluteLogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    out = zeros(size(x, 1), size(x, 1))

    # If we are in the wrong place, choose a gradient of 0 and hope constraints pull it back out.
    if _violates_bound(c, x)
        return out
    end

    # Set the hessian at one entry since the cost is independent of other costs.
    out[c.idx, c.idx] = (x[c.idx] - c.offset)^(-2)
    return out
end

function Guus(c::AbsoluteLogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(us)
    return Dict(jj => zeros(size(us[jj], 1), size(us[jj], 1)) for jj in 1:num_players)
end

# Export the derivatives.
export Gx, Gus, Gxx, Guus

# Export all the cost type.
export AbsoluteLogBarrierCost

# Export all the cost types/structs and functionality.
export compute_cost
