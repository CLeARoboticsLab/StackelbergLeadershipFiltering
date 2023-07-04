using Plots

include("LQ_parameters.jl")

Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, ss_costs, T, leader_idx)

xs, us = unroll_feedback(dyn, times, Ps_strategies, x₁)

final_cost_totals = [evaluate(ss_costs[ii], xs, us) for ii in 1:num_players]
println("final: ", xs[:, T], " with trajectory costs: ", final_cost_totals, " sum: ", sum(final_cost_totals))
println(size(xs), " ", size(us[1]), " ", size(us[2]))


using ElectronDisplay

# Plot positions, other two states, controls, and convergence.
q = @layout [a b; c d ;e f; g h]

q1, q2, q3, q4, q5, q6, q7 = plot_states_and_controls(dyn, times, xs, us)

plot(q1, q2, q3, q4, q5, q6, q7, layout = q)
