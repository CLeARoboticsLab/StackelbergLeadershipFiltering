# Define an example cost for iLQR

struct ExampleILQRCost <: NonQuadraticCost
    quad_cost::QuadraticCost
    c::Float64 # constant
    max_accel::Float64
    max_omega::Float64
    x_final::AbstractVector{<:Float64}
    is_nonlinear::Bool
end

# See slide 35 on Jake Levy's CLeAR Lab talk on 09/26/2022.
function compute_cost(c::ExampleILQRCost, time_range, xh::AbstractVector{Float64}, uhs::AbstractVector{<:AbstractVector{Float64}})
    out_size = size(xh, 1)
    xhf = homogenize_vector(c.x_final)

    dxh_tt = xh - xhf

    # Start with the quadratic portions of the cost.
    total = compute_cost(c.quad_cost, time_range, dxh_tt, uhs)

    accel = uhs[1][2]
    omega = uhs[1][1]
    if c.is_nonlinear
        total += exp(c.c * (accel^2 - c.max_accel^2))
        total += exp(c.c * (omega^2 - c.max_omega^2))
    end

    return total
end

function quadraticize_costs(cost::ExampleILQRCost, time_range, x, us)

    accel = us[1][2]
    omega = us[1][1]
    c = cost.c

    # Create the affine cost.
    aff_cost = QuadraticCost(cost.quad_cost.Q)

    R = 2 * cost.quad_cost.Rs[1]

    if cost.is_nonlinear
        # This is a bug because it computes the whole cost including state cost.
        xh = homogenize_vector(x)
        uhs = homogenize_ctrls(cost, us)
        const_term = compute_cost(cost, time_range, xh, uhs)
        exp_term_alpha = exp(c * (accel^2 - cost.max_accel^2))
        exp_term_omega = exp(c * (omega^2 - cost.max_omega^2))
        r = 2 * c * [accel * exp_term_alpha;  omega * exp_term_omega]
        R += Diagonal([2*c * exp_term_alpha * (1 + 2 * c * accel^2),
                       2*c * exp_term_omega * (1 + 2 * c * omega^2)])

        add_control_cost!(aff_cost, 1, R; r, const_term)
    else
        add_control_cost!(aff_cost, 1, R)
    end
    return quadraticize_costs(aff_cost, time_range, x, us)
end

export ExampleILQRCost, quadraticize_costs, compute_cost
