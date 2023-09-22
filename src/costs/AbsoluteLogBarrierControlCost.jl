# This is a 1D barrier on control values, independent of other dimensions.
struct AbsoluteLogBarrierControlCost <: NonQuadraticCost
    player_idx::Int           # index of the player controls that are to be costed
    a::Vector{Float64}        # a vector that is multiplied by state to determine cost
    offset::Real              # scalar offset
    is_lower_bound::Bool      # true indicates lower bound barrier, false indicates upper bound
end

const LARGE_NUMBER = 1e6

function get_as_function(c::AbsoluteLogBarrierControlCost)
    f(si, x, us, t) = begin
        input = _get_input(c, us)
        if _violates_bound(c, us)
            return LARGE_NUMBER + 100*(input-0.01)^2 # input should be negative if bounds violated, this introduces a slope.
        end
        return -log(input)
    end
end
export get_as_function


# Compute input to log function based on whether we have an upper or lower bound.
function _get_input(c::AbsoluteLogBarrierControlCost, us)
    return (c.is_lower_bound) ? c.a' * us[c.player_idx] - c.offset : c.offset - c.a' * us[c.player_idx]
end

function _violates_bound(c::AbsoluteLogBarrierControlCost, us)
    violates_lower_bound = c.is_lower_bound && c.a' * us[c.player_idx] ≤ c.offset
    violates_upper_bound = !c.is_lower_bound && c.a' * us[c.player_idx] ≥ c.offset
    return violates_upper_bound || violates_lower_bound
end

function compute_cost(c::AbsoluteLogBarrierControlCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    if _violates_bound(c, us)
        return LARGE_NUMBER
    else
        input = _get_input(c, us)
        return -log(input)
    end
end

# Derivative term c(x_i - a_i)^-1 at {c.idx}th index.
function Gx(c::AbsoluteLogBarrierControlCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return zeros(1, size(x, 1))
end

function Gus(c::AbsoluteLogBarrierControlCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
    num_players = length(us)
    return Dict(jj => zeros(1, size(us[jj], 1)) for jj in 1:num_players)
end

# diagonal matrix, with diagonals either 0 or c(x_i - a_i)^-2, no cross terms
function Gxx(c::AbsoluteLogBarrierControlCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    out = zeros(size(x, 1), size(x, 1))
    return out

    # # If we are in the wrong place, choose a gradient of 0 and hope constraints pull it back out.
    # if _violates_bound(c, x)
    #     return out
    # end

    # # Set the hessian at one entry since the cost is independent of other costs.
    # out[c.idx, c.idx] = (x[c.idx] - c.offset)^(-2)
    # return out
end

function Guus(c::AbsoluteLogBarrierControlCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
    num_players = length(us)
    return Dict(jj => zeros(size(us[jj], 1), size(us[jj], 1)) for jj in 1:num_players)
end

# Export the derivatives.
export Gx, Gus, Gxx, Guus

# Export all the cost type.
export AbsoluteLogBarrierControlCost

# Export all the cost types/structs and functionality.
export compute_cost
