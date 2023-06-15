# This struct defines a quadratic tracking cost about some reference state and controls.

struct QuadraticTrackingCost <: Cost
    "This quadratic cost is defined on arguments dx = x - x_ref and dus = us - us_ref as inputs."
    q_cost::QuadraticCost
    xr
    urs
end
make_quadratic_tracking_cost(quad_cost::QuadraticCost, xr=zeros(size(quad_cost.Q, 1)), urs=[zeros(size(quad_cost.Rs[ii], 1)) for ii in 1:length(quad_cost.Rs)]) = begin
    @assert all(iszero.(quad_cost.x₀)) "Quadratic cost provided to QuadraticTrackingCost ctor must have no state offset."
    @assert all([all(iszero.(quad_cost.u₀s[ii])) for ii in 1:length(quad_cost.u₀s)]) "Quadratic cost provided to QuadraticTrackingCost ctor must have no control offsets."
    return QuadraticTrackingCost(quad_cost, xr, urs)
end

# Constructors for shifted input (x-a, u^i - b^i) quadratic costs.
# Shifts a (Q, q, cq) quadratic cost set to transform to (Q̃, q̃, c̃q). Can be used for control sets too.
function _shift_quad_cost(Q, q, cq, a)
    @assert size(q) == size(a)
    Q̃ = deepcopy(Q)
    q̃ = q - Q * a
    c̃q = (1/2) * (a' * Q * a) - (q' * a) + cq

    return Q̃, q̃, c̃q
end

# TODO(hmzh): Adjust quadraticization for the multi-player case.
function quadraticize_costs(c::QuadraticTrackingCost, time_range, x, us)
    # return quadraticize_costs(deepcopy(c.q_cost), time_range, x - c.xr, us - c.urs)

    # # 1. Get and shift the parameters of the quadratic cost.
    Q = get_quadratic_state_cost_term(c.q_cost)
    q = get_linear_state_cost_term(c.q_cost)
    cq = get_constant_state_cost_term(c.q_cost)
    # Q_new, q_new, cq_new = _shift_quad_cost(Q, q, cq, c.xr)

    # # 2. Create a new intermediary quadratic cost with these and fill with control costs too.
    # c_new = QuadraticCost(Q_new, q_new, cq_new)
    # for ii in 1:length(us)
    #     R = get_quadratic_control_cost_term(c.q_cost, ii)
    #     r = get_linear_control_cost_term(c.q_cost, ii)
    #     cr = get_constant_control_cost_term(c.q_cost, ii)
    #     R_new, r_new, cr_new = _shift_quad_cost(R, r, cr, c.urs[ii])
    #     add_control_cost!(c_new, ii, R_new; r=r_new, cr=cr_new)
    # end

    # # 3. Repeat the same offset, for now.
    # add_offsets!(c_new, c.q_cost.x₀, c.q_cost.u₀s)

    # return c_new

    a = c.xr - x
    q = q - Q * a
    cq = (1/2.) * (a' * Q * a) - (q' * a) + cq

    # We need to split this cost over multiple matrices.
    # num_mats = length(c.q_cost.Rs) + 1
    # cq = (1.0 / num_mats) * c.xr' * Q * c.xr
    # cr = (1.0 / num_mats) * c.xr' * Q * c.xr

    cost = QuadraticCost(Q, q, cq)
    for (ii, R) in c.q_cost.Rs
        R = get_quadratic_control_cost_term(c.q_cost, ii)
        r = get_linear_control_cost_term(c.q_cost, ii)
        cr = get_constant_control_cost_term(c.q_cost, ii)

        b = c.urs[ii] - us[ii]
        r = r - R * b
        cr = (1/2.) * (b' * R * b) - (r' * b) + cr

        add_control_cost!(cost, ii, R; r, cr)
    end
    add_offsets!(cost, zeros(4), [zeros(size(c.q_cost.Rs[ii], 1)) for ii in 1:length(c.q_cost.Rs)])
    # add_offsets!(cost, c.xr-x, c.urs - us)
    # println(q)
    # add_offsets!(cost, x - c.xr, us - c.urs)

    return cost
end

# Computes a cost recentered about the x offset and us offset.
function compute_cost(c::QuadraticTrackingCost, time_range, x, us)
    dx = x - c.xr
    dus = us - c.urs
    return compute_cost(c.q_cost, time_range, dx, dus)
end

# Define derivative terms.
function Gx(c::QuadraticTrackingCost, time_range, x, us)
    Q = get_quadratic_state_cost_term(c.q_cost)
    q = get_linear_state_cost_term(c.q_cost)
    return (x - c.xr)' * Q + q'
end

function Gus(c::QuadraticTrackingCost, time_range, x, us)
    return Dict(ii => (us[ii] - c.urs[ii])' * R + c.q_cost.rs[ii]' for (ii, R) in c.q_cost.Rs)
end

function Gxx(c::QuadraticTrackingCost, time_range, x, us)
    return get_quadratic_state_cost_term(c.q_cost)
end

function Guus(c::QuadraticTrackingCost, time_range, x, us)
    return deepcopy(c.q_cost.Rs)
end


# Export all the cost type.
export QuadraticTrackingCost, make_quadratic_tracking_cost

# Export all the cost types/structs and functionality.
export add_control_cost!, quadraticize_costs, compute_cost

# Export derivative terms
export Gx, Gus, Gxx, Guus
