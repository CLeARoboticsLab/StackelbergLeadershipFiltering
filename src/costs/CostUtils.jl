# Utilities for managing quadratic and nonquadratic dynamics.
abstract type Cost end

# Every Cost is assumed to have the following functions defined on it:
# - quadraticize_costs(cost, time_range, x, us) - this function produces a PureQuadraticCost at time t given the state and controls
# - compute_cost(cost, xs, us) - this function evaluates the cost of a trajectory given the states and controls
# - Gx, Gus, Gxx, Guus(cost, time_range, x, us) - derivatives must be defined.

# We use this type by making a substruct of it which can then have certain functions defined for it.
abstract type NonQuadraticCost <: Cost end

# Evaluates the cost across a time horizon.
# - xs[:, time]
# - us[player][:, time]
function evaluate(c::Cost, xs, us)
    horizon = last(size(xs))
    num_players = size(us, 1)

    total = 0.0
    for tt in 1:horizon
        prev_time = (tt == 1) ? tt : tt-1
        us_tt = [us[ii][:, tt] for ii in 1:num_players]
        total += compute_cost(c, (prev_time, tt), xs[:, tt], us_tt)
    end
    return total
end

function quadraticize_costs(c::Cost, time_range, x, us)
    num_players = length(us)

    cost_eval = compute_cost(c, time_range, x, us)
    ddx2 = Gxx(c, time_range, x, us)
    dx = Gx(c, time_range, x, us)
    ddu2s = Guus(c, time_range, x, us)
    dus = Gus(c, time_range, x, us)

    # Used to compute the way the constant cost terms are divided.
    num_cost_mats = length(ddu2s)
    const_cost_term = (2/num_cost_mats) * cost_eval

    # This should be QuadraticCost with offset about x because the taylor approx is (x-x0)
    quad_cost = QuadraticCost(ddx2, dx', const_cost_term)
    for (ii, du) in dus
        add_control_cost!(quad_cost, ii, ddu2s[ii]; r=dus[ii]', cr=const_cost_term)
    end

    return quad_cost
end


# Export all the cost types/structs.
export Cost, NonQuadraticCost, evaluate
