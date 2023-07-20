using StackelbergControlHypothesesFiltering

using LinearAlgebra

include("LQ_parameters.jl")

# costs = [QuadraticCostWithOffset(ss_costs[1]), QuadraticCostWithOffset(ss_costs[2])]
# costs = ss_costs

# For player cost
costs = [pc_cost_1, pc_cost_2]

num_runs=1

# config variables
threshold=1e-2
max_iters=1000
step_size=1e-2
verbose=true

sg_obj = initialize_silq_games_object(num_runs, T, dyn, costs;
                                      state_reg_param=0.0, control_reg_param=1e-32, ensure_pd=false,
                                      threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times, x₁, us_1)

println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals, " sum: ", sum(final_cost_totals))
println(size(xs_k), " ", size(us_k[1]), " ", size(us_k[2]))


using ElectronDisplay
using Plots

# Plot positions, other two states, controls, and convergence.
# q = @layout [a b c; d e f; g h i]
p = @layout grid(3, 1)
plot_title = "SILQGames on LQ Game"

q1, q2, q3, q4, q5, q6, q7 = plot_states_and_controls(dyn, times, xs_k, us_k)


q8, q9 = plot_convergence_and_costs(num_iters, threshold, conv_metrics, evaluated_costs)

# plot(q1, q2, q3, q4, q5, q6, q7, q8, q9, layout = q)
plot(q1, q8, q9, layout=p, plot_title=plot_title)


plot!(q1, title="", legend=:bottomleft, xaxis=[-2.5, 2.5], yaxis=[-2.5, 2.5], legendfontsize = 11, tickfontsize=11, fontsize=11)
filename = string("silq_lq_results_leader", leader_idx, "_3_position.pdf")
savefig(q1, filename)

plot!(q8, title="", xaxis=[-0.1, 1.1], xticks=[0, 1], legendfontsize = 11, tickfontsize=11, fontsize=11)
filename = string("silq_lq_results_leader", leader_idx, "_3_convergence.pdf")
savefig(q8, filename)

plot!(q9, title="", xaxis=[-0.1, 1.1], xticks=[0, 1], legendfontsize = 11, tickfontsize=11, fontsize=11)
filename = string("silq_lq_results_leader", leader_idx, "_3_cost.pdf")
savefig(q9, filename)



