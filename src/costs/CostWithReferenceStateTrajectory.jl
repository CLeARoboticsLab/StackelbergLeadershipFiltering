# Affine costs with quadratic, linear, constant terms.
# Should always be used when there is a need for Taylor approximations of non-LQ systems.

mutable struct CostWithReferenceStateTrajectory <: Cost
    cost::Cost
    xrefs::AbstractVector{Float64}
    t₀::Float64
end
CostWithReferenceStateTrajectory(cost, xrefs) = CostWithReferenceStateTrajectory(cost, xrefs, 0.)

function get_as_function(c::CostWithReferenceStateTrajectory)
    f(si, x, us, t) = begin


        dx = x - c.x0
        dus = us - c.u0s

        terms = c.q_cost
        cost = (1//2) * (dx' * terms.Q * dx) + (dx' * terms.q) + terms.cq
        cost = cost + (1//2) * dus[1]' * terms.Rs[1] * dus[1] + (dus[1]' * terms.rs[1]) + terms.crs[1]
        cost = cost + (1//2) * dus[2]' * terms.Rs[2] * dus[2] + (dus[2]' * terms.rs[2]) + terms.crs[2]
        return cost
    end
    return f
end
export get_as_function


# TODO(hmzh): Adjust quadraticization for the multi-player case.
function quadraticize_costs(c::QuadraticCostWithOffset, time_range, x, us)
    Q = get_quadratic_state_cost_term(c.q_cost)
    q = get_linear_state_cost_term(c.q_cost)

    q = Q * (x - c.x0) + q

    # We need to split this cost over multiple matrices.
    num_mats = length(c.q_cost.Rs) + 1
    cq = (1.0 / num_mats) * c.x0' * Q * c.x0
    cr = (1.0 / num_mats) * c.x0' * Q * c.x0

    cost = QuadraticCost(Q, q, cq)

    for (ii, R) in c.q_cost.Rs
        add_control_cost!(cost, ii, c.q_cost.Rs[ii]; r=c.q_cost.Rs[ii] * (us[ii] - c.u0s[ii]) + c.q_cost.rs[ii], cr=cr)
    end

    return cost
end

function compute_cost(c::CostWithReferenceStateTrajectory, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end

# Derivative term c(x_i - a_i)^-1 at {c.idx}th index.
function Gx(c::CostWithReferenceStateTrajectory, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end

function Gus(c::CostWithReferenceStateTrajectory, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end

# diagonal matrix, with diagonals either 0 or c(x_i - a_i)^-2, no cross terms
function Gxx(c::CostWithReferenceStateTrajectory, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end

function Guus(c::CostWithReferenceStateTrajectory, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    error("not implemented")
end


# Export all the cost type.
export QuadraticCostWithOffset

# Export all the cost types/structs and functionality.
export add_control_cost!, quadraticize_costs, compute_cost

# Export derivative terms
export Gx, Gus, Gxx, Guus
