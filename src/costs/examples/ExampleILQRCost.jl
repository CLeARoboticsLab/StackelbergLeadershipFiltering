# Define an example cost for iLQR which considers unicycle dynamics.
# cost = (x-xf)' Q (x-xf) + u' R u + exp(omega^2 - omega^2_max) + exp(accel^2 - accel^2_max)

struct ExampleILQRCost <: NonQuadraticCost
    quad_cost::QuadraticCostWithOffset
    c::Float64 # constant
    max_accel::Float64
    max_omega::Float64
    x_final::AbstractVector{<:Float64}
    is_nonlinear::Bool
end

# See slide 35 on Jake Levy's CLeAR Lab talk on 09/26/2022.
function compute_cost(c::ExampleILQRCost, time_range, x, us)
    # Start with the quadratic portions of the c.
    total = compute_cost(c.quad_cost, time_range, x, us)

    accel = us[1][2]
    omega = us[1][1]
    if c.is_nonlinear
        total += exp(c.c * (accel^2 - c.max_accel^2))
        total += exp(c.c * (omega^2 - c.max_omega^2))
    end

    return total
end

function Gx(c::ExampleILQRCost, time_range, x, us)
    return Gx(c.quad_cost, time_range, x, us)
end

function Gus(c::ExampleILQRCost, time_range, x, us)
    accel = us[1][2]
    omega = us[1][1]
    exp_term_alpha = exp(c.c * (accel^2 - c.max_accel^2))
    exp_term_omega = exp(c.c * (omega^2 - c.max_omega^2))

    out = Gus(c.quad_cost, time_range, x, us)
    if c.is_nonlinear
        out[1] += 2 * c.c * [omega * exp_term_omega; accel * exp_term_alpha]'
    end
    return out
end

function Gxx(c::ExampleILQRCost, time_range, x, us)
    return Gxx(c.quad_cost, time_range, x, us)
end

function Guus(c::ExampleILQRCost, time_range, x, us)
    accel = us[1][2]
    omega = us[1][1]
    exp_term_alpha = exp(c.c * (accel^2 - c.max_accel^2))
    exp_term_omega = exp(c.c * (omega^2 - c.max_omega^2))

    out = Guus(c.quad_cost, time_range, x, us)
    if c.is_nonlinear
        out[1] += 2 * c.c * Diagonal([(1 + 2 * omega^2) * exp_term_omega, (1 + 2 * accel^2) * exp_term_alpha])
    end
    return out
end

export ExampleILQRCost, compute_cost, Gx, Gus, Gxx, Guus
