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

stackelberg_leader_idx = 1

function coupling_example()
    
    # Dynamics (Euler-discretized double integrator equations with Δt = 0.1s).
    # State for each player is layed out as [x, ẋ, y, ẏ].
    Ã = [1 0.1 0 0;
     0 1   0 0;
     0 0   1 0.1;
     0 0   0 1]
    A = vcat(hcat(Ã, zeros(4, 4)), hcat(zeros(4, 4), Ã))

    B₁ = vcat([0   0;
           0.1 0;
           0   0;
           0   0.1],
          zeros(4, 2)
          )
    B₂ = vcat(zeros(4, 2),
          [0   0;
           0.1 0;
           0   0;
           0   0.1]
          )
    dyn = Dynamics(A, [B₁, B₂])

    # Costs reflecting the preferences above.
    Q₁ = zeros(8, 8)
    Q₁[5, 5] = 1.0
    Q₁[7, 7] = 1.0
    c₁ = Cost(Q₁)
    add_control_cost!(c₁, 1, 0.1 * diagm([1, 1]))

    Q₂ = zeros(8, 8)
    Q₂[1, 1] = 1.0
    Q₂[5, 5] = 1.0
    Q₂[1, 5] = -1.0
    Q₂[5, 1] = -1.0
    Q₂[3, 3] = 1.0
    Q₂[7, 7] = 1.0
    Q₂[3, 7] = -1.0
    Q₂[7, 3] = -1.0
    c₂ = Cost(Q₂)
    add_control_cost!(c₂, 2, 0.1 * diagm([1, 1]))

    costs = [c₁, c₂]

    # Initial condition chosen randomly. Ensure both have relatively low speed.
    x₁ = randn(8)
    x₁[[2, 4, 6, 8]] .= 0

    # Solve over a horizon of 100 timesteps.
    horizon = 50

    Ps = solve_lq_nash_feedback(dyn, costs, horizon)
    xs_nash_feedback, us_nash_feedback = unroll_feedback(dyn, Ps, x₁)
    Ls = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
    # println(Ls[stackelberg_leader_idx])
    xs_stackelberg_feedback, us_stackelberg_feedback = unroll_feedback(dyn, Ls, x₁)
    # println(xs_stackelberg_feedback)
    # println(us_stackelberg_feedback)

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
