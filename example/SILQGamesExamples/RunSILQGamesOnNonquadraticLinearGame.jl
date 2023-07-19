using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Statistics

# includes linear dynamics and quadratic costs
include("nonquadratic_linear_parameters.jl")

num_runs=1

# config variables
threshold=1e-3
max_iters=2000
step_size=1e-2
verbose=true

sg_obj = initialize_silq_games_object(num_runs, T, dyn, costs;
                                      threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times, x₁, us_1)

println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals)
println(size(xs_k), " ", size(us_k[1]), " ", size(us_k[2]))


using ElectronDisplay
using Plots

# Plot positions, other two states, controls, and convergence.
# q = @layout [a b; c d; e f]
# q = @layout [a b; c d]#; e f]



q1, q2, q3, q4, q5, q6, q7 = plot_states_and_controls(dyn, times, xs_k, us_k)


# Plot convergence and costs.
q8, q9 = plot_convergence_and_costs(num_iters, threshold, conv_metrics, evaluated_costs)


# plot(q1, q2, q3, q4, q5, q6, q7, q8, q9, layout = q)


plot!(q1, title="", legend=:bottomleft, xaxis=[-2.5, 2.5], yaxis=[-2.5, 2.5], legendfontsize = 11, tickfontsize=11, fontsize=11)
filename = string("silq_lnq_bound_2_1_results_leader", leader_idx, "_3_position.pdf")
savefig(q1, filename)

plot!(q8, title="", legendfontsize = 14)
filename = string("silq_lnq_bound_2_1_results_leader", leader_idx, "_3_convergence.pdf")
savefig(q8, filename)

plot!(q9, title="", legendfontsize = 14)
filename = string("silq_lnq_bound_2_1_results_leader", leader_idx, "_3_cost.pdf")
savefig(q9, filename)



