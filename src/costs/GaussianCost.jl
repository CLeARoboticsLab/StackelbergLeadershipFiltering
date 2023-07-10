# A cost that has a "bump" at a provided (sub)state.
struct GaussianCost <: NonQuadraticCost
    x_idx::Vector{Int}
    μ # mean
    Σ_inv # covariance_inv
    is_impl::Bool
end
GaussianCost(x_idx, μ, Σ) = begin
    @assert isposdef(Σ)
    GaussianCost(x_idx, μ, inv(Σ), false)
end

function get_as_function(c::GaussianCost)
    n = size(c.μ, 1)
    Σ_det = 1/det(c.Σ_inv)
    coeff = 1/sqrt((2*pi)^n * Σ_det)
    f(si, x, us, t) = coeff * exp(-(1//2) * (x[c.x_idx]-c.μ)' * c.Σ_inv * (x[c.x_idx]-c.μ))
    return f
end
export get_as_function


function compute_cost(c::GaussianCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end

# Derivative term c(x_i - a_i)^-1 at {c.idx}th index.
function Gx(c::GaussianCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end

function Gus(c::GaussianCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end

# diagonal matrix, with diagonals either 0 or c(x_i - a_i)^-2, no cross terms
function Gxx(c::GaussianCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end

function Guus(c::GaussianCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end

# Export the derivatives.
export Gx, Gus, Gxx, Guus

# Export all the cost type.
export GaussianCost

# Export all the cost types/structs and functionality.
export compute_cost
