# Affine costs with quadratic, linear, constant terms.
# Should always be used when there is a need for Taylor approximations of non-LQ systems.

struct QuadraticCostWithOffset <: Cost
    q_cost::QuadraticCost
    x0::AbstractVector{Float64}
    u0s::AbstractVector{<:AbstractVector{Float64}}
end
QuadraticCostWithOffset(quad_cost::QuadraticCost, x0=zeros(size(quad_cost.Q, 1)), u0s=[zeros(size(quad_cost.Rs[ii], 1)) for ii in 1:length(quad_cost.Rs)]) = QuadraticCostWithOffset(quad_cost, x0, u0s)


function get_as_function(c::QuadraticCostWithOffset)
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

function compute_cost(c::QuadraticCostWithOffset, time_range, x, us)
    dx = x - c.x0
    dus = us - c.u0s
    return compute_cost(c.q_cost, time_range, dx, dus)
end

# Define derivative terms.
function Gx(c::QuadraticCostWithOffset, time_range, x, us)
    Q = get_quadratic_state_cost_term(c.q_cost)
    q = get_linear_state_cost_term(c.q_cost)
    return (x - c.x0)' * Q + q'
end

function Gus(c::QuadraticCostWithOffset, time_range, x, us)
    return Dict(ii => (us[ii] - c.u0s[ii])' * R + c.q_cost.rs[ii]' for (ii, R) in c.q_cost.Rs)
end

function Gxx(c::QuadraticCostWithOffset, time_range, x, us)
    return get_quadratic_state_cost_term(c.q_cost)
end

function Guus(c::QuadraticCostWithOffset, time_range, x, us)
    return deepcopy(c.q_cost.Rs)
end


# Export all the cost type.
export QuadraticCostWithOffset

# Export all the cost types/structs and functionality.
export add_control_cost!, quadraticize_costs, compute_cost

# Export derivative terms
export Gx, Gus, Gxx, Guus
