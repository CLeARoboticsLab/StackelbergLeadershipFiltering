# Affine costs with quadratic, linear, constant terms.
# Should always be used when there is a need for Taylor approximations of non-LQ systems.

struct QuadraticCostWithOffset <: Cost
    q_cost::QuadraticCost
    x0::AbstractVector{Float64}
    u0s::AbstractVector{<:AbstractVector{Float64}}
end
# Quadratic costs always homogeneous
QuadraticCostWithOffset(quad_cost::QuadraticCost, x0=zeros(size(quad_cost.Q, 1)), u0s=[zeros(size(quad_cost.Rs[ii], 1)) for ii in 1:length(quad_cost.Rs)]) = QuadraticCostWithOffset(quad_cost, x0, u0s)

# # Shifts a (Q, q, cq) quadratic cost set to transform to (Q̃, q̃, c̃q). Can be used for control sets too.
# function shift_cost(Q, q, cq, a)
#     @assert size(q) == size(a)
#     Q̃ = Q
#     q̃ = q - Q * a
#     c̃q = (1/2) * (a' * Q * a) - (q' * a) + cq

#     return Q̃, q̃, c̃q
# end

# # one helpful use case is designing a cost at the origin x=0 and then shifting it elsewhere.
# # a is the state shift x-a
# # bs define the control shifts u^i - b^i
# function make_shifted_quadratic_cost(c::QuadraticCost, a, bs)
#     Q = get_quadratic_state_cost_term(c.q_cost)
#     q = get_linear_state_cost_term(c.q_cost)
#     cq = get_constant_state_cost_term(c.q_cost)

#     Q̃, q̃, c̃q = shift_cost(Q, q, cq)
#     cost = QuadraticCost(Q̃, q̃, c̃q, a)
#     for ii in 1:size(bs, 1)
#         Ri = get_quadratic_control_cost_term(c.q_cost, ii)
#         ri = get_linear_control_cost_term(c.q_cost, ii)
#         cri = get_constant_control_cost_term(c.q_cost, ii)
#         R̃, r̃, c̃r = shift_cost(Ri, ri, cri, bs[ii])
#         add_control_cost!(cost, R̃; r=r̃, cr=c̃r)
#     end
#     return cost
# end



# TODO(hmzh): Adjust quadraticization for the multi-player case.
function quadraticize_costs(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    Q = get_quadratic_state_cost_term(c.q_cost)
    q = get_linear_state_cost_term(c.q_cost)

    # TODO(hamzah) - need to recenter the quadraticization about the new x0, u0s
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

function compute_cost(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    dx = x - c.x0
    return compute_cost(c.q_cost, time_range, dx, us)
end

# Define derivative terms.
function Gx(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    Q = get_quadratic_state_cost_term(c.q_cost)
    q = get_linear_state_cost_term(c.q_cost)
    return (x - c.x0)' * Q + q'
end

function Gus(c::QuadraticCostWithOffset, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return Dict(ii => (us[ii] - c.u0s[ii])' * R + c.q_cost.rs[ii]' for (ii, R) in c.q_cost.Rs)
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
