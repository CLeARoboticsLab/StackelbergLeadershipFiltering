# Define an example cost for iLQR

struct ExampleILQRCost <: NonQuadraticCost
    Q # quadtratic state cost
    R # quadratic control cost
    c # constant
    max_accel::Float64
    max_omega::Float64
    x_final::AbstractVector{<:Float64}
end

# See slide 35 on Jake Levy's CLeAR Lab talk on 09/26/2022.
function evaluate(c::ExampleILQRCost, xs, us)
    horizon = last(size(xs))

    # If system is homogenized, adjust the vector.
    xf = (length(c.x_final) + 1 == size(xs, 1)) ? vcat(c.x_final, [1]) : c.x_final

    total = 0.0
    for tt in 1:horizon
        dx_tt = xs[:, tt] - xf
        total += dx_tt' * c.Q * dx_tt

        total += us[1][:, tt]' * c.R * us[1][:, tt]

        accel = us[1][2, tt]
        omega = us[1][1, tt]
        total += exp(c.c * (accel^2 - c.max_accel^2))
        total += exp(c.c * (omega^2 - c.max_omega^2))
    end
    return total
end

function quadraticize_costs(cost::ExampleILQRCost, time_range, x, us)

    accel = us[1][2]
    omega = us[1][1]

    q_cost = QuadraticCost(2 * cost.Q)
    c = cost.c

    cr = evaluate(cost, x[:, :], [u[:, :] for u in us])
    exp_term_alpha = exp(c*(accel^2 - cost.max_accel^2))
    exp_term_omega = exp(c*(omega^2 - cost.max_omega^2))

    r = 2*c * [accel * exp_term_alpha  omega * exp_term_omega]
    R = 2 * cost.R + Diagonal([2*c * exp_term_alpha * (1 + 2*c*accel^2),
                               2*c * exp_term_omega * (1 + 2*c*omega^2)])

    R̃ = homogenize_cost(R, r[:], cr)
    add_control_cost!(q_cost, 1, R̃)

    return q_cost
end

export ExampleILQRCost, quadraticize_costs, evaluate
