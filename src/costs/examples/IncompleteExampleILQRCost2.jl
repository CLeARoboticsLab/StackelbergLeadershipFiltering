# This cost can be described as follows;.
# - some quadratic cost x'Ix + exp(x1^2(t) + x2^2(t)) + <quadratic control costs>
#   first derivative is [2*x1]


# still uses the quadratics cost terms
struct NonlinearILQRCost <: NonQuadraticCost
    player_idx::Int
    quad_cost::QuadraticCost
end

function get_player_state(c::NonlinearILQRCost, x)
    ii = cost.player_idx
    x1, x2, x3, x4 = x[4*(ii-1)+1:4*ii]
    return x1, x2, x3, x4
end

# Define the nonlinear cost computation and two derivatives with respect to x and u.
function compute_cost(c::NonlinearILQRCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(x) == 8
    @assert length(us[1]) == length(us[2]) == 2
    u1 = us[1]
    u2 = us[2]
    x1, x2, x3, x4 = get_player_state(cost, x)
    return ln(x1) + exp(x2) + x3*x4 + u1' * u1 + u1' * u2 + u2' * u2

     = quad_cost
end

function Gx(c::NonlinearILQRCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(x) == 9
    x1, x2, x3, x4 = get_player_state(cost, x)
    return [1/x1 exp(x2) x4 x3]
end

function Gxx(c::NonlinearILQRCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(x) == 9
    Z2 = zeros(2, 2)
    x1, x2, x3, x4 = get_player_state(cost, x)
    return vcat(hcat(diagm([-x1^(-2), exp(x2)]), Z2),
                hcat(                       Z2 , [0 1; 1 0]))
end

function Gus(c::NonlinearILQRCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(us[1]) == length(us[2]) == 2
    return [2*us[1] +   us[2],
              us[1] + 2*us[2]]
end

function Guus(c::NonlinearILQRCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(us[1]) == length(us[2]) == 2

    num_u = length(us[1])
    return ones(num_u, num_states) + I
end

export compute_cost, Gx, Gxx, Gus, Guus
