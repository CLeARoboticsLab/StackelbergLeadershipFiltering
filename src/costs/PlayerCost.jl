# A player cost defined by an arbitrary, twice differentiable function f(x, us, t).

struct PlayerCost <: NonQuadraticCost
    f::Function      # Takes inputs f(c.si, x, us, t).
    si::SystemInfo   # Information about the game being played.
end

function get_as_function(c::PlayerCost)
    return c.f
end
export get_as_function

function compute_cost(c::PlayerCost, time_range, x, us)
    t = time_range[2]
    return c.f(c.si, x, us, t)
end

# Derivative term wrt state.
function Gx(c::PlayerCost, time_range, x0, u0s)
    t = time_range[2]
    x_cost = x -> c.f(c.si, x, u0s, t)
    ForwardDiff.gradient(x_cost, x0)
end

function Gus(c::PlayerCost, time_range, x0, u0s)
    t = time_range[2]
    u0s_combined = vcat(u0s...)
    diff_us = us -> c.f(c.si, x0, split(us, udims(c.si)), t)
    dus = ForwardDiff.gradient(diff_us, u0s_combined)

    # Split the gradient into segments for each actor.
    dus_dict = Dict()
    idx = 1
    for ii in 1:num_agents(c.si)
        dus_dict[ii] = dus[idx:idx+udim(c.si, ii)-1]
        idx = idx + udim(c.si, ii)
    end
    return dus_dict
end

function Gxx(c::PlayerCost, time_range, x0, u0s)
    t = time_range[2]
    x_cost = x -> c.f(c.si, x, u0s, t)

    result = DiffResults.HessianResult(x0)
    result = ForwardDiff.hessian!(result, x_cost, x0)
    return DiffResults.hessian(result)
end

function Guus(c::PlayerCost, time_range, x0, u0s)
    t = time_range[2]
    u0s_combined = vcat(u0s...)
    diff_us = us -> c.f(c.si, x0, split(us, udims(c.si)), t)

    result = DiffResults.HessianResult(u0s_combined)
    result = ForwardDiff.hessian!(result, diff_us, u0s_combined)
    Duu = DiffResults.hessian(result)

    # Split the hessian block matrices into segments for each actor. Ignore the others for now.
    duus_dict = Dict()
    idx = 1
    for ii in 1:num_agents(c.si)
        duus_dict[ii] = Duu[idx:idx+udim(c.si, ii)-1, idx:idx+udim(c.si, ii)-1]
        idx = idx + udim(c.si, ii)
    end
    return duus_dict
end

# Export the derivatives.
export Gx, Gus, Gxx, Guus

# Export all the cost type.
export PlayerCost

# Export all the cost types/structs and functionality.
export compute_cost
