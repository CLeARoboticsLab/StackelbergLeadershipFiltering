# Affine costs with quadratic, linear, constant terms.
mutable struct QuadraticCost <: Cost
    Q
    q
    cq
    Rs
    rs
    crs
    x₀
    u₀s
end
QuadraticCost(Q::AbstractMatrix{<:Real}) = QuadraticCost(Q, zeros(size(Q, 1)), 0,
                                                          Dict{Int, Matrix{eltype(Q)}}(),
                                                          Dict{Int, Vector{eltype(Q)}}(),
                                                          Dict{Int, eltype(Q)}(),
                                                          nothing,
                                                          nothing)
QuadraticCost(Q::AbstractMatrix{<:Real}, q::AbstractVector{<:Real}, cq::Real) = QuadraticCost(Q, q, cq,
                                                                                          Dict{Int, Matrix{eltype(Q)}}(),
                                                                                          Dict{Int, Vector{eltype(q)}}(),
                                                                                          Dict{Int, eltype(cq)}(),
                                                                                          nothing,
                                                                                          nothing)

function add_control_cost!(c::QuadraticCost, other_player_idx, R; r=zeros(size(R, 1)), cr=0.)
    @assert size(R, 1) == size(R, 2) == size(r, 1)
    @assert size(cr) == ()

    c.Rs[other_player_idx] = R
    c.rs[other_player_idx] = r
    c.crs[other_player_idx] = cr

end

function add_offsets!(c::QuadraticCost, x₀, u₀s)
    @assert isnothing(c.x₀)
    @assert isnothing(c.u₀s)

    num_players = length(u₀s)
    @assert size(c.Q, 1) == length(x₀)
    @assert all([size(c.Rs[ii], 1) == length(u₀s[ii]) for ii in 1:num_players])

    c.x₀ = x₀
    c.u₀s = u₀s
end

# If the offsets are unset, then we set them to 0 by default.
function _set_zero_offsets_if_unset(c::QuadraticCost, xsize, usizes)
    # Both offsets should be identically nothing or identically set.
    @assert isnothing(c.x₀) == isnothing(c.u₀s)

    if isnothing(c.x₀)
        c.x₀ = zeros(xsize)
        c.u₀s = [zeros(usizes[ii]) for ii in 1:length(usizes)]
    end
end

function compute_cost(c::QuadraticCost, t, x, us)
    _set_zero_offsets_if_unset(c, size(x), [size(us[ii]) for ii in 1:length(us)])

    num_players = length(us)
    dx = x - c.x₀
    dus = us - c.u₀s
    total = ((1/2.) * dx' * c.Q * dx) + c.q' * dx + c.cq
    if !isempty(c.Rs)
        total += sum(((1/2.) * dus[jj]' * R * dus[jj]) + dus[jj]' * c.rs[jj] + c.crs[jj] for (jj, R) in c.Rs)
    end
    return total
end


# Constructors for shifted input (x-a, u^i - b^i) quadratic costs.
# Shifts a (Q, q, cq) quadratic cost set to transform to (Q̃, q̃, c̃q). Can be used for control sets too.
function _shift_cost(Q, q, cq, a)
    @assert size(q) == size(a)
    Q̃ = deepcopy(Q)
    q̃ = q - Q * a
    c̃q = (1/2) * (a' * Q * a) - (q' * a) + cq

    return Q̃, q̃, c̃q
end

# TODO(hamzah) - simplify this and potentially get rid of offsets altogether, but not in this PR.
function quadraticize_costs(c::QuadraticCost, time_range, x, us)
    _set_zero_offsets_if_unset(c, size(x), [size(us[ii]) for ii in 1:length(us)])

    num_players = length(us)

    # 1. Extract the quadratic cost in standard vector quadratic form, i.e. s.t. an equivalent
    #    g(x) = (1/2) * x' * Q_new * x + x' * q_new + cq_new
    #    Note: This form does not have the (x - c.x₀) terms as these are factored into the new values of Q, q, cq.
    # Q_new = c.Q
    # q_new = c.q - c.Q * c.x₀
    # cq_new = (1./2) * c.x₀' * c.Q * c.x₀ - c.q' * c.x₀ + c.cq
    Q_new, q_new, cq_new = _shift_cost(c.Q, c.q, c.cq, c.x₀)

    # 2. Create a new intermediary quadratic cost with these and fill with control costs too.
    c_new = QuadraticCost(Q_new, q_new, cq_new)
    for ii in 1:length(us)
        R_new, r_new, cr_new = _shift_cost(c.Rs[ii], c.rs[ii], c.crs[ii], c.u₀s[ii])
        add_control_cost!(c_new, ii, R_new; r=r_new, cr=cr_new)
    end
    # These have zero offsets, so set them.
    _set_zero_offsets_if_unset(c_new, size(x), [size(us[ii]) for ii in 1:length(us)])

    # 3. Use the new cost we defined to compute new constants which have the same collective cost mass as the original.
    cost_eval = compute_cost(c_new, time_range, x, us)
    cq = cr = (1/(num_players+1)) * cost_eval

    # TODO(hamzah): Compute these gradients/Hessians using ForwardDiff. The previous code is still necessary for
    #               consistent constants. Constants don't need to be consistent, but they are helpful for testing.
    #               This code can then be moved to CostUtils.jl as a generalized implementation of quadraticization.
    # ddx2 = Gxx(c, time_range, x, us)
    # dx = Gx(c, time_range, x, us)
    # ddu2s = Guus(c, time_range, x, us)
    # dus = Gus(c, time_range, x, us)

    # 4. Use the new values when manually computing the gradients and Hessians.
    ddx2 = c_new.Q
    ddu2s = [c_new.Rs[jj] for jj in 1:num_players]

    dx = x' * c_new.Q + c_new.q'
    dus = [us[jj]' * c_new.Rs[jj] + c_new.rs[jj]' for jj in 1:num_players]

    quad_cost = QuadraticCost(ddx2, dx', cq)
    for jj in 1:num_players
        add_control_cost!(quad_cost, jj, ddu2s[jj]; r=dus[jj]', cr)
    end
    add_offsets!(quad_cost, x, us)

    return quad_cost
end


# Derivative terms
function Gx(c::QuadraticCost, t, x, us)
    return x' * c.Q + c.q'
end

function Gus(c::QuadraticCost, t, x, us)
    return Dict(ii => us[ii]' * R + c.rs[ii]' for (ii, R) in c.Rs)
end

function Gxx(c::QuadraticCost, t, x, us)
    return deepcopy(c.Q)
end

function Guus(c::QuadraticCost, t, x, us)
    return deepcopy(c.Rs)
end

export Gx, Gus, Gxx, Guus


# Helpers specific to quadratic costs - these return the cost terms in the form
# x' Q x + q' x + cq.
# Offsets are factored into the terms.
# TODO(hamzah) - Find a way to define offsets more automatically.
function get_quadratic_state_cost_term(c::QuadraticCost)
    @assert !isnothing(c.x₀) "Offsets must be defined manually."
    Q_shifted = _shift_cost(c.Q, c.q, c.cq, c.x₀)[1]
    return Q_shifted
end

function get_linear_state_cost_term(c::QuadraticCost)
    @assert !isnothing(c.x₀) "Offsets must be defined manually."
    q_shifted = _shift_cost(c.Q, c.q, c.cq, c.x₀)[2]
    return q_shifted
end

function get_constant_state_cost_term(c::QuadraticCost)
    @assert !isnothing(c.x₀) "Offsets must be defined manually."
    cq_shifted = _shift_cost(c.Q, c.q, c.cq, c.x₀)[3]
    return cq_shifted
end

function get_quadratic_control_cost_term(c::QuadraticCost, player_idx::Int)
    @assert !isnothing(c.u₀s) "Offsets must be defined manually."
    R_shifted = _shift_cost(c.Rs[player_idx], c.rs[player_idx], c.crs[player_idx], c.u₀s[player_idx])[1]
    return R_shifted
end

function get_linear_control_cost_term(c::QuadraticCost, player_idx::Int)
    @assert !isnothing(c.u₀s) "Offsets must be defined manually."
    r_shifted = _shift_cost(c.Rs[player_idx], c.rs[player_idx], c.crs[player_idx], c.u₀s[player_idx])[2]
    return r_shifted
end

function get_constant_control_cost_term(c::QuadraticCost, player_idx::Int)
    @assert !isnothing(c.u₀s) "Offsets must be defined manually."
    cr_shifted = _shift_cost(c.Rs[player_idx], c.rs[player_idx], c.crs[player_idx], c.u₀s[player_idx])[3]
    return cr_shifted
end

# Export the cost type.
export QuadraticCost

# Export the helpers.
export get_quadratic_state_cost_term, get_linear_state_cost_term, get_constant_state_cost_term,
       get_quadratic_control_cost_term, get_linear_control_cost_term, get_constant_control_cost_term

# Export all the cost types/structs and functionality.
export add_control_cost!, add_offsets!, quadraticize_costs, compute_cost
