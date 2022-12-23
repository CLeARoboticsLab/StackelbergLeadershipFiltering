# This game can be described as follows.
# - nonlinear unicycle dynamics
# - nonquadratic state cost x'Ix + exp(x1^2(t) + x2^2(t)) + <quadratic control costs>
#   first derivative is [2*x1]


# still uses the quadratics cost terms
struct ExampleNonQuadraticCost <: NonQuadraticCost
    player_idx::Int
end

function get_player_state(cost::ExampleNonQuadraticCost, x)
    ii = cost.player_idx
    x1, x2, x3, x4 = x[4*(ii-1)+1:4*ii]
    return x1, x2, x3, x4
end

# Define the nonlinear cost computation and two derivatives with respect to x and u.
function g(cost::ExampleNonQuadraticCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(x) == 9
    @assert length(us[1]) == length(us[2]) == 2
    u1 = us[1]
    u2 = us[2]
    x1, x2, x3, x4 = get_player_state(cost, x)
    return ln(x1) + exp(x2) + x3*x4 + u1' * u1 + u1' * u2 + u2' * u2
end

function Gx(cost::ExampleNonQuadraticCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(x) == 9
    x1, x2, x3, x4 = get_player_state(cost, x)
    return [1/x1 exp(x2) x4 x3]
end

function Gxx(cost::ExampleNonQuadraticCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(x) == 9
    Z2 = zeros(2, 2)
    x1, x2, x3, x4 = get_player_state(cost, x)
    return vcat(hcat(diagm([-x1^(-2), exp(x2)]), Z2),
                hcat(                       Z2 , [0 1; 1 0]))
end

function Gu(cost::ExampleNonQuadraticCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(us[1]) == length(us[2]) == 2
    return [2*us[1] +   us[2],
              us[1] + 2*us[2]]
end

function Guu(cost::ExampleNonQuadraticCost, tt, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    @assert length(us[1]) == length(us[2]) == 2

    num_u = length(us[1])
    return ones(num_u, num_states) + I
end

function quadraticize_costs(cost::ExampleNonQuadraticCost, tt, x0::AbstractVector{Float64}, u0s::AbstractVector{<:AbstractArray{Float64}})
    num_players = length(u0s)
    @assert length(x0) == 9

    # Linearize about reference state and control trajectories x0, u0s.
    G_x0x0 = Gxx(tt, x0, u0s)
    G_x0 = Gx(tt, x0, u0s)
    g_0 = g(tt, x0, u0s)

    Q = G_x0x0
    q = G_x0 + x0' * G_x0x0
    cq = g_0 + 0.5 * x0' * G_x0x0 * x0 - G_x0 * x0

    # Construct the quadratic cost and add in the linearized control costs.
    cost = AffineCost(Q̃, q, cq)

    for jj in 1:num_players
        u0 = u0s[jj]

        G_u0u0 = Gxx(tt, x0, u0s)
        G_u0 = Gx(tt, x0, u0s)
        g_0 = g(tt, x0, u0s)

        R = G_u0u0
        r = G_u0 + u0' * G_u0u0
        cr = g_0 + 0.5 * u0' * G_u0x0 * u0 - G_u0 * u0

        add_control_cost!(cost, jj, R, r, cr)
    end

    # Turn affine cost into quadratic.
    return quadraticize_costs(cost, tt, x0, u0s)
end

# Evaluate cost on a state/control trajectory.
# - xs[:, time]
# - us[player][:, time]
function evaluate(c::ExampleNonQuadraticCost, xs, us)
    horizon = last(size(xs))
    N = length(us)

    total = 0.0
    for tt in 1:horizon
        us_tt = [us[jj][:, tt] for jj in 1:N]
        total += g(tt, xs[:, tt], us_tt)
    end
    return total
end



using StackelbergControlHypothesesFiltering
using LinearAlgebra: diagm

# Script to compare nash and stackelberg feedback solutions to a test problem involving
# coupling between two players.
# Game is described as follows:
#   - both players' dynamics are decoupled and follow double integrator motion
#     in the Cartesian plane
#   - P1 wants *P2* to get to the origin
#   - P2 wants to get close to *P1*
#   - both players want to expend minimal control effort

stackelberg_leader_idx = 2

function non_lq_example()

    num_players = 2
    dyn = UnicycleDynamics(num_players)
    costs = [ExampleNonQuadraticCost(1), ExampleNonQuadraticCost(2)]

    # Initial condition chosen randomly. Ensure both have relatively low speed. These are homogenized
    num_states = xdim(dyn.sys_info) + 1
    x₁ = randn(num_states)
    x₁[[2, 4, 6, 8]] .= 0
    x₁[9] = 1

    # Solve over a horizon of 100 timesteps, noting that it's not good for the system to leave 0.
    horizon = 100
    t0 = 0
    x_refs = vcat(zeros(num_states-1, horizon), ones(1, horizon))
    u_refs = [zeros(udim(dyn, ii), horizon) for ii in 1:num_players]

    Ps, _ = solve_approximated_lq_nash_feedback(dyn, costs, horizon, t0, x_refs, u_refs)
    xs_nash_feedback, us_nash_feedback = unroll_feedback(dyn, FeedbackGainControlStrategy(Ps), x₁)

    Ss, _ = solve_approximated_lq_stackelberg_feedback(dyn, costs, horizon, t0, x_refs, u_refs, stackelberg_leader_idx)
    xs_stackelberg_feedback, us_stackelberg_feedback = unroll_feedback(dyn, FeedbackGainControlStrategy(Ss), x₁)

    return xs_nash_feedback, us_nash_feedback, xs_stackelberg_feedback, us_stackelberg_feedback
end

# Call this fn.
xs_nash_feedback, us_nash_feedback, xs_stackelberg_feedback, us_stackelberg_feedback = non_lq_example()
horizon = last(size(xs_nash_feedback))

# Plot.
using ElectronDisplay
using Plots

p = plot(legend=:outertopright)

# Nash feedback.
α = 0.2
plot!(p, xs_nash_feedback[1, :], xs_nash_feedback[3, :],
      seriestype=:scatter, arrow=true, seriescolor=:blue, label="P1 Nash feedback")
plot!(p, xs_nash_feedback[1, :], xs_nash_feedback[3, :],
      seriestype=:quiver,  seriescolor=:blue,
      quiver=(0.1 * xs_nash_feedback[2, :], 0.1 * xs_nash_feedback[4, :]),
      seriesalpha=α,
      label="P1 vel (Nash FB)")
plot!(p, xs_nash_feedback[1, :], xs_nash_feedback[3, :],
      seriestype=:quiver,  seriescolor=:turquoise,
      quiver=(0.1 * us_nash_feedback[1][1, :], 0.1 * us_nash_feedback[1][2, :]),
      seriesalpha=α,
      label="P1 acc (Nash FB)")

plot!(p, xs_nash_feedback[5, :], xs_nash_feedback[7, :],
      seriestype=:scatter, arrow=true, seriescolor=:red, label="P2 Nash feedback")
plot!(p, xs_nash_feedback[5, :], xs_nash_feedback[7, :],
      seriestype=:quiver,  seriescolor=:red,
      quiver=(0.1 * xs_nash_feedback[6, :], 0.1 * xs_nash_feedback[8, :]),
      seriesalpha=α,
      label="P2 vel (Nash FB)")
plot!(p, xs_nash_feedback[5, :], xs_nash_feedback[7, :],
      seriestype=:quiver,  seriescolor=:pink,
      quiver=(0.1 * us_nash_feedback[2][1, :], 0.1 * us_nash_feedback[2][2, :]),
      seriesalpha=α,
      label="P2 acc (Nash FB)")

# Stackelberg feedback.
p2_label = (stackelberg_leader_idx == 2) ? "P2 (leader) Stackelberg Feedback" : "P2 (follower) Stackelberg Feedback"
plot!(p, xs_stackelberg_feedback[5, :], xs_stackelberg_feedback[7, :],
      seriestype=:scatter, arrow=true, seriescolor=:purple, label=p2_label)
plot!(p, xs_stackelberg_feedback[5, :], xs_stackelberg_feedback[7, :],
      seriestype=:quiver,  seriescolor=:purple,
      quiver=(0.1 * xs_stackelberg_feedback[6, :], 0.1 * xs_stackelberg_feedback[8, :]),
      seriesalpha=α,
      label="P2 vel (Stackelberg FB)")
plot!(p, xs_stackelberg_feedback[5, :], xs_stackelberg_feedback[7, :],
      seriestype=:quiver,  seriescolor=:magenta,
      quiver=(0.1 * us_stackelberg_feedback[2][1, :], 0.1 * us_stackelberg_feedback[2][2, :]),
      seriesalpha=α,
      label="P2 acc (Stackelberg FB)")

p1_label = (stackelberg_leader_idx == 1) ? "P1 (leader) Stackelberg Feedback" : "P1 (follower) Stackelberg Feedback"
plot!(p, xs_stackelberg_feedback[1, :], xs_stackelberg_feedback[3, :],
      seriestype=:scatter, arrow=true, seriescolor=:green, label=p1_label)
plot!(p, xs_stackelberg_feedback[1, :], xs_stackelberg_feedback[3, :],
      seriestype=:quiver,  seriescolor=:green,
      quiver=(0.1 * xs_stackelberg_feedback[2, :], 0.1 * xs_stackelberg_feedback[4, :]),
      seriesalpha=α,
      label="P1 vel (Stackelberg FB)")
plot!(p, xs_stackelberg_feedback[1, :], xs_stackelberg_feedback[3, :],
      seriestype=:quiver,  seriescolor=:lightgreen,
      quiver=(0.1 * us_stackelberg_feedback[1][1, :], 0.1 * us_stackelberg_feedback[1][2, :]),
      seriesalpha=α,
      label="P1 acc (Stackelberg FB)")

display(p)
