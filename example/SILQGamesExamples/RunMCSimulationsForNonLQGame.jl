# NOTE: Remember to change the cost of the LQ shepherd and sheep game.

using StackelbergControlHypothesesFiltering

using LinearAlgebra
using ProgressBars
using Statistics
using StatsBase

include("nonLQ_parameters.jl")
include("SILQGamesMCUtils.jl")

num_sims = 20
num_buckets = 8

# config variables
mc_threshold=1e-3
mc_max_iters=2000
mc_step_size=1e-2
mc_verbose=false

sg_obj = initialize_silq_games_object(num_sims, T, dyn, costs;
                                      threshold=mc_threshold, max_iters=mc_max_iters, step_size=mc_step_size, verbose=mc_verbose)

# Nominal trajectory is always zero-controls. x₁ is drawn as follows: P1 starts at (2, 1) unmoving and P2 rotates in a circle about the origin at the same radius.
x1 = x₁[xidx(dyn, 1)]
y1 = x₁[yidx(dyn, 1)]

p1_angle = atan(y1, x1)
p1_magnitude = norm([x1, y1])

# Run MC sims.
sim_iters = ProgressBar(1:num_sims)
for iter in sim_iters
    new_x₁, new_us_1 = get_initial_conditions_at_idx(dyn, iter, num_sims, p1_angle, p1_magnitude, x₁)
    # new_us_1 = [vcat(zeros(1, T), 0.05*ones(1, T)) for ii in 1:num_players]
    xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times, new_x₁, new_us_1)
    # println("$iter - Converged status (", is_converged, ") after ", num_iters, " iterations.")
    # if num_iters > 2
    #     final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
    #     println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals, " sum: ", sum(final_cost_totals))
    #     println("final cost: ", final_cost_totals, " sum: ", sum(final_cost_totals))
    # end
end


using ElectronDisplay
using Plots


# 1. Plot convergence metric max absolute state difference between iterations.

# Configure the convergence plot.
convergence_plot = plot_convergence(sg_obj; lower_bound=mc_threshold/10)

# 2. Plot the convergence histogram.
convergence_histogram = plot_convergence_histogram(sg_obj, num_buckets)


# trajectory distance to origin (for each player)
d1 = plot_distance_to_origin(dyn, sg_obj)

# trajectory distance between agents
d2 = plot_distance_to_agents(dyn, sg_obj)


plot!(convergence_plot, title="")
filename = string("silq_nonlq_mc$(num_sims)_convergence_L$(leader_idx).pdf")
savefig(convergence_plot, filename)

plot!(convergence_histogram, title="", legend=:none, tickfontsize=16, fontsize=12, labelfontsize=14)
filename = string("silq_nonlq_mc$(num_sims)_convhistogram_L$(leader_idx).pdf")
savefig(convergence_histogram, filename)

plot!(d1, title="")
filename = string("silq_nonlq_mc$(num_sims)_L$(leader_idx)_dist_to_origin.pdf")
savefig(d1, filename)

plot!(d2, title="")
filename = string("silq_nonlq_mc$(num_sims)_L$(leader_idx)_dist_to_agent.pdf")
savefig(d2, filename)


# l = @layout [a b; c d]
# plot(convergence_plot, convergence_histogram, d1, d2, layout=l)


# Generate a gif to see results.
iter = ProgressBar(1:num_sims)
anim = @animate for k in iter
    p = @layout [a b c; d e f; g h i]

    xns = sg_obj.xks[k, :, :]
    un1s = sg_obj.uks[1][k, :, :]
    un2s = sg_obj.uks[2][k, :, :]

    p1, p2, p3, p4, p5, p6, p7 = plot_states_and_controls(dyn, times, xns, [un1s, un2s])
    plot!(p1, xlimits=(-2.5, 2.5), ylimits=(-2.5, 2.5))

    # Plot convergence.
    num_iters = sg_obj.num_iterations[k]
    conv_x = cumsum(ones(num_iters)) .- 1
    conv_metrics = sg_obj.convergence_metrics[k, :, :]
    evaluated_costs = sg_obj.evaluated_costs[k, :, :]

    r1 = plot(conv_x, conv_metrics[1, 1:num_iters], title="conv.", label="p1", yaxis=:log)
    # plot!(r1, conv_x, conv_metrics[2, 1:num_iters], label="p2", yaxis=:log)
    plot!(r1, [k, k], [minimum(conv_metrics[1, 1:num_iters]), maximum(conv_metrics[1, 1:num_iters])], label="", color=:black, yaxis=:log)

    # r2 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label="p1", yaxis=:log)
    # plot!(r2, conv_x, evaluated_costs[2, 1:num_iters], label="p2", yaxis=:log)
    # plot!(r2, [k, k], [minimum(evaluated_costs[:, 1:num_iters]), maximum(evaluated_costs[:, 1:num_iters])], label="", color=:black, yaxis=:log)

    # Shift the cost to ensure they are positive.
    costs_1 = evaluated_costs[1, 1:num_iters] .+ (abs(minimum(evaluated_costs[1, 1:num_iters])) + 1e-8)
    costs_2 = evaluated_costs[2, 1:num_iters] .+ (abs(minimum(evaluated_costs[2, 1:num_iters])) + 1e-8)

    q6 = plot(conv_x, costs_1, title="evaluated costs", label="p1", yaxis=:log)
    plot!(q6, conv_x, costs_2, label="p2", yaxis=:log)
    plot!(q6, [k, k], [minimum(costs_1), maximum(costs_2)], label="", color=:black, yaxis=:log)

    plot(p1, p2, p3, p4, p5, p6, p7, r1, q6, layout = p)
end

# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
gif(anim, "mc$(num_sims)_nonlq_silqgames_animation.gif", fps = 5)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
