# A script that takes as input a JLD file containing S SILQGames trajectories and makes cumulative and individual plots.
using StackelbergControlHypothesesFiltering

# using LinearAlgebra
using ProgressBars
# using Statistics
# using StatsBase

using Dates
# using Distributions
using LaTeXStrings
# using Random
using Plots

using JLD

include("MCFileLocations.jl")
include("SILQGamesMCUtils.jl")

mc_silq_foldername = joinpath(mc_folder, data_folder)
mc_silq_jld_name = silq_data_file
input_data_path = joinpath(mc_silq_foldername, mc_silq_jld_name)

derivative_plots_folder = "plots_$(get_date_str())"
plots_path = joinpath(mc_silq_foldername, derivative_plots_folder)
isdir(plots_path) || mkdir(plots_path)

# num_buckets = 

data = load(input_data_path)["data"]

dyn = data["dyn"]
T = data["T"]
times = data["times"][1:T]

num_sims = data["num_sims"]
all_xs = data["xs"]
all_us = data["us"]

leader_idx = data["gt_leader_idx"]
mc_threshold = data["threshold"]
mc_max_iters = data["max_iters"]

all_num_iters = data["num_iterations"]
all_convergence_metrics = data["convergence_metrics"]
all_evaluated_costs = data["evaluated_costs"]

for iter in ProgressBar(1:num_sims)
    folder_name = joinpath(plots_path, string(iter))
    isdir(folder_name) || mkdir(folder_name)

    xs_k = all_xs[iter, :, :]
    us_k = [@view all_us[jj][iter, :, :] for jj in 1:num_agents(dyn)]

    q1, _, _, _, _, _, _ = plot_states_and_controls(dyn, times, xs_k, us_k; include_legend=:outertop)

    conv_metrics = all_convergence_metrics[iter, :, :]
    evaluated_costs = all_evaluated_costs[iter, :, :]
    q8, q9 = plot_convergence_and_costs(all_num_iters[iter], mc_threshold, conv_metrics, evaluated_costs)

    plot!(q1, title="", linewidth=10)
    filepath = joinpath(folder_name, "$(iter)_silq_position_L$(leader_idx).pdf")
    savefig(q1, filepath)

    plot!(q8, title="")
    filepath = joinpath(folder_name, "$(iter)_silq_convergence_L$(leader_idx).pdf")
    savefig(q8, filepath)

    plot!(q9, title="")
    filepath = joinpath(folder_name, "$(iter)_silq_cost_L$(leader_idx).pdf")
    savefig(q9, filepath)
end


############################
#### Make the MC plots. ####
############################
# 1. Plot convergence metric max absolute state difference between iterations. For the LQ case, we should only need 1 iteration.
convergence_plot = plot_convergence(all_convergence_metrics, all_num_iters, mc_max_iters, mc_threshold; lower_bound=1e-6)

# 2. Plot the convergence histogram. For the LQ case, all items should be in first bin.
convergence_histogram = plot_convergence_histogram(all_num_iters)

# trajectory distance to origin (for each player)
d1 = plot_distance_to_origin(dyn, times, all_xs)

# trajectory distance between agents
d2 = plot_distance_to_agents(dyn, times, all_xs)


###########################
#### Save the figures. ####
###########################
plot!(convergence_plot, title="")
filename = joinpath(plots_path, string("silq_mc$(num_sims)_convergence_L$(leader_idx).pdf"))
savefig(convergence_plot, filename)

plot!(convergence_histogram, title="", tickfontsize=16, fontsize=12, labelfontsize=14)
filename = joinpath(plots_path, string("silq_mc$(num_sims)_convhistogram_L$(leader_idx).pdf"))
savefig(convergence_histogram, filename)

plot!(d1, title="")
filename = joinpath(plots_path, string("silq_mc$(num_sims)_L$(leader_idx)_dist_to_origin.pdf"))
savefig(d1, filename)

plot!(d2, title="")
filename = joinpath(plots_path, string("silq_mc$(num_sims)_L$(leader_idx)_dist_to_agent.pdf"))
savefig(d2, filename)


#############################
#### Make the debug gif. ####
#############################


# Generate a gif to see results.
iter = ProgressBar(1:num_sims)
anim = @animate for k in iter
    p = @layout [a b c; d e f; g h i]

    xs = all_xs[k, :, :]
    u1s = all_us[1][k, :, :]
    u2s = all_us[2][k, :, :]

    p1, p2, p3, p4, p5, p6, p7 = plot_states_and_controls(dyn, times, xs, [u1s, u2s])
    plot!(p1, xlimits=(-2.5, 2.5), ylimits=(-2.5, 2.5))

    # Plot convergence.
    num_iters = all_num_iters[k]
    conv_x = cumsum(ones(num_iters)) .- 1
    conv_metrics = all_convergence_metrics[k, :, :]
    evaluated_costs = all_evaluated_costs[k, :, :]

    r1 = plot(conv_x, conv_metrics[1, 1:num_iters], title="conv.", label=L"\mathcal{A}_1", yaxis=:log)
    # plot!(r1, conv_x, conv_metrics[2, 1:num_iters], label=L"\mathcal{A}_2", yaxis=:log)
    plot!(r1, [k, k], [minimum(conv_metrics[1, 1:num_iters]), maximum(conv_metrics[1, 1:num_iters])], label="", color=:black, yaxis=:log)

    # r2 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label=L"\mathcal{A}_1", yaxis=:log)
    # plot!(r2, conv_x, evaluated_costs[2, 1:num_iters], label=L"\mathcal{A}_2", yaxis=:log)
    # plot!(r2, [k, k], [minimum(evaluated_costs[:, 1:num_iters]), maximum(evaluated_costs[:, 1:num_iters])], label="", color=:black, yaxis=:log)

    # Shift the cost to ensure they are positive.
    costs_1 = evaluated_costs[1, 1:num_iters] .+ (abs(minimum(evaluated_costs[1, 1:num_iters])) + 1e-8)
    costs_2 = evaluated_costs[2, 1:num_iters] .+ (abs(minimum(evaluated_costs[2, 1:num_iters])) + 1e-8)

    q6 = plot(conv_x, costs_1, title="evaluated costs", label=L"\mathcal{A}_1", yaxis=:log)
    plot!(q6, conv_x, costs_2, label=L"\mathcal{A}_2", yaxis=:log)
    plot!(q6, [k, k], [minimum(costs_1), maximum(costs_2)], label="", color=:black, yaxis=:log)

    plot(p1, p2, p3, p4, p5, p6, p7, r1, q6, layout = p)
end

# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
gif(anim, joinpath(plots_path, "silqgames_animation.gif"), fps = 5)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype



