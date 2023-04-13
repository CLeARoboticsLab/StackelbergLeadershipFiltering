# A cost c*log((b-a)/(norm_fn(x-a))), where norm_fn reduces the state to a scalar.
# a, b, c shape the log function, a shifts the function right, b selects the x-intercept, c is a scaling factor
# indices select the state components to compute the norm over
# Assumes no control cost
struct LogBarrierCost <: NonQuadraticCost
    indices
    a::Vector{Float64} # offset with size identical to x
    b::Float64         # sets the x-intercept
    c::Float64         # scaling constant
end
LogBarrierCost(indices, a::Vector{Float64}, b::Float64) = LogBarrierCost(indices, a, b, 1.)

# Use the 1 norm for now.
P_NORM = 1

function compute_cost(c::LogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert size(x, 1) == size(c.a, 1)
    abs_dx = norm(x[c.indices] - c.a[c.indices], P_NORM)
    return -log(abs_dx) + log(c.b)
end

# Derivative term c(x_i - a_i)^-1
function Gx(c::LogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert size(x, 1) == size(c.a, 1)
    out = zeros(1, size(x, 1))
    dx = x[c.indices] - c.a[c.indices]
    out[1, c.indices] .= - c.c ./ dx
    return out
end

function Gus(c::LogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(us)
    return Dict(jj => zeros(1, size(us[jj], 1)) for jj in 1:num_players)
end

# diagonal matrix, with diagonals either 0 or c(x_i - a_i)^-2, no cross terms
function Gxx(c::LogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert size(x, 1) == size(c.a, 1)
    out = zeros(size(x, 1), size(x, 1))
    for i in c.indices
        out[i, i] = c.c / ((x[i] - c.a[i])^2)
    end
    return out
end

function Guus(c::LogBarrierCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    num_players = length(us)
    return Dict(jj => zeros(size(us[jj], 1), size(us[jj], 1)) for jj in 1:num_players)
end

# Export the derivatives.
export Gx, Gus, Gxx, Guus

# Export all the cost type.
export LogBarrierCost

# Export all the cost types/structs and functionality.
export compute_cost
