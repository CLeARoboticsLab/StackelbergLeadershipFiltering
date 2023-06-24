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

function coupling_example()

    dt = 0.1
    cont_lin_dyn = ShepherdAndSheepDynamics()
    dyn = discretize(cont_lin_dyn, dt)
    costs = ShepherdAndSheepCosts(dyn)

    # Initial condition chosen randomly. Ensure both have relatively low speed.
    x₁ = randn(8)
    x₁[[2, 4, 6, 8]] .= 0

    # Solve over a horizon of 100 timesteps.
    horizon = 100

    ctrl_strats_nash, _ = solve_lq_nash_feedback(dyn, costs, horizon)
    times = cumsum(ones(horizon)) .- 1.
    xs_nash_feedback, us_nash_feedback = unroll_feedback(dyn, times, ctrl_strats_nash, x₁)
    ctrl_strats_stack, _ = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
    xs_stackelberg_feedback, us_stackelberg_feedback = unroll_feedback(dyn, times, ctrl_strats_stack, x₁)

    return xs_nash_feedback, us_nash_feedback, xs_stackelberg_feedback, us_stackelberg_feedback
end

# Call this fn.
xs_nash_feedback, us_nash_feedback, xs_stackelberg_feedback, us_stackelberg_feedback = coupling_example()
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
