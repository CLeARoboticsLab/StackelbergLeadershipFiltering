# NOTE: Remember to change the cost of the LQ shepherd and sheep game.

using StackelbergControlHypothesesFiltering

using LinearAlgebra
using ProgressBars
using Statistics
using StatsBase

using Distributions
using LaTeXStrings
using Random
using Plots

include("quadratic_nonlinear_parameters.jl")
include("SILQGamesMCUtils.jl")

num_sims = 4
num_buckets = 4
# costs = [QuadraticCostWithOffset(costs[1]), QuadraticCostWithOffset(costs[2])]
costs = [pc_cost_1, pc_cost_2]
# costs = ss_costs

topfolder_name = "uq_mc$(num_sims)_L$(leader_idx)_$(Dates.now())"
isdir(topfolder_name) || mkdir(topfolder_name)

# config variables
mc_threshold=4e-3
mc_max_iters=1000
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
# Threads.@threads for iter in sim_iters
ii = 0
for iter in sim_iters
    global ii += 1
    folder_name = joinpath(topfolder_name, string(ii))
    isdir(folder_name) || mkdir(folder_name)

    new_x₁, new_us_1 = get_initial_conditions_at_idx(dyn, iter, num_sims, p1_angle, p1_magnitude, x₁)
    xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times, new_x₁, new_us_1; manual_idx=iter)

    q1, _, _, _, _, _, _ = plot_states_and_controls(dyn, times, xs_k, us_k; include_legend=:outertop)
    q8, q9 = plot_convergence_and_costs(num_iters, mc_threshold, conv_metrics, evaluated_costs)

    plot!(q1, title="", xaxis=[-2.5, 2.5], yaxis=[-2.5, 2.5])
    filepath = joinpath(folder_name, "$(ii)_silq_uq_position_L$(leader_idx).pdf")
    # filename = string("silq_uq_results_leader", leader_idx, "_3_position.pdf")
    savefig(q1, filepath)

    plot!(q8, title="")
    # filename = string("silq_uq_results_leader", leader_idx, "_3_convergence.pdf")
    filepath = joinpath(folder_name, "$(ii)_silq_uq_convergence_L$(leader_idx).pdf")
    savefig(q8, filepath)

    plot!(q9, title="")
    # filename = string("silq_uq_results_leader", leader_idx, "_3_cost.pdf")
    filepath = joinpath(folder_name, "$(ii)_silq_uq_cost_L$(leader_idx).pdf")
    savefig(q9, filepath)

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
filename = joinpath(topfolder_name, string("silq_uq_mc$(num_sims)_convergence_L$(leader_idx).pdf"))
savefig(convergence_plot, filename)

plot!(convergence_histogram, title="", tickfontsize=16, fontsize=12, labelfontsize=14)
filename = joinpath(topfolder_name, string("silq_uq_mc$(num_sims)_convhistogram_L$(leader_idx).pdf"))
savefig(convergence_histogram, filename)

plot!(d1, title="")
filename = joinpath(topfolder_name, string("silq_uq_mc$(num_sims)_L$(leader_idx)_dist_to_origin.pdf"))
savefig(d1, filename)

plot!(d2, title="")
filename = joinpath(topfolder_name, string("silq_uq_mc$(num_sims)_L$(leader_idx)_dist_to_agent.pdf"))
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
gif(anim, "mc$(num_sims)_uq_silqgames_animation.gif", fps = 5)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype




# Leadership filtering.
t0 = times[1]
lf_times = dt * (cumsum(ones(2*T)) .- 1)
pos_unc = 1e-3
θ_inc = 1e-3
vel_unc = 1e-4
P₁ = Diagonal([pos_unc, pos_unc, θ_inc, vel_unc, pos_unc, pos_unc, θ_inc, vel_unc])

# Process noise uncertainty
Q = 1e-1 * Diagonal([1e-2, 1e-2, 1e-3, 1e-4, 1e-2, 1e-2, 1e-3, 1e-4])

# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = 0.05 * Matrix(I, xdim(dyn), xdim(dyn))
Ts = 30
num_games = 1
num_particles = 50

p_transition = 0.98
p_init = 0.5

discrete_state_transition, state_trans_P = generate_discrete_state_transition(p_transition, p_transition)
s_init_distrib = Bernoulli(p_init)

process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q)

lf_threshold = 1e-3
lf_max_iters = 50
lf_step_size = 1e-2

all_probs = zeros(num_sims, T)

sim_iters = ProgressBar(1:num_sims)
ii = 0
for ss in sim_iters
    global ii += 1
    folder_name = joinpath(topfolder_name, string(ii))
    isdir(folder_name) || mkdir(folder_name)

    # Extract states and controls from simulation.
    true_xs = sg_obj.xks[ss, :, :]
    true_us = [sg_obj.uks[ii][ss, :, :] for ii in 1:num_players]

    # Augment the remaining states so we have T+Ts-1 of them.
    xs = hcat(true_xs, zeros(xdim(dyn), Ts-1))
    us = [hcat(true_us[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

    zs = zeros(xdim(dyn), T)

    # Fill in z as noisy state measurements.
    for tt in 1:T
        zs[:, tt] = rand(rng, MvNormal(true_xs[:, tt], R))
    end

    # println("LF start from x1: ", xs[:, 1], zs[:, 1])

    x̂s, P̂s, all_probs[ss, :], pf, sg_objs = leadership_filter(dyn, costs, t0, lf_times,
                           T,         # simulation horizon
                           Ts,        # horizon over which the stackelberg game should be played,
                           num_games, # number of stackelberg games played for measurement
                           xs[:, 1],  # initial state at the beginning of simulation
                           P₁,        # initial covariance at the beginning of simulation
                           us,        # the control inputs that the actor takes
                           zs,        # the measurements
                           R,
                           process_noise_distribution,
                           s_init_distrib,
                           discrete_state_transition;
                           threshold=lf_threshold,
                           rng,
                           max_iters=lf_max_iters,
                           step_size=lf_step_size,
                           Ns=num_particles,
                           verbose=false)

    

    # Only needs to be generated once.
    p1a = plot_leadership_filter_positions(sg_objs[1].dyn, true_xs[:, 1:T], x̂s[:, 1:T], zs[:, 1:T])

    pos_main_filepath = joinpath(folder_name, "lf_uq_positions_main_L$(leader_idx).pdf")
    savefig(p1a, pos_main_filepath)
    snapshot_freq = Int((T - 1)/10)
    jj = 0
    for t in 2:snapshot_freq:T
        jj += 1
        plot!(p1a, legend=:bottomleft)

        p1b = plot_leadership_filter_measurement_details(num_particles, sg_objs[t], true_xs[:, 1:T], x̂s)

        p5_jj, p6_jj = make_probability_plots(times[1:T], all_probs[ss, 1:T]; t_idx=t, include_gt=leader_idx)
        plot!(p5_jj, title="")
        plot!(p6_jj, title="")

        pos2_filepath = joinpath(folder_name, "0$(jj)_lf_t$(t)_uq_positions_detail_L$(leader_idx).pdf")
        prob1_filepath = joinpath(folder_name, "0$(jj)_lf_t$(t)_uq_probs_P1_L$(leader_idx).pdf")
        prob2_filepath = joinpath(folder_name, "0$(jj)_lf_t$(t)_uq_probs_P2_L$(leader_idx).pdf")
        
        savefig(p1b, pos2_filepath)
        savefig(p5_jj, prob1_filepath)
        savefig(p6_jj, prob2_filepath)
    end
end

mean_probs = mean(all_probs, dims=[1])[1, :]
stddev_probs = (size(all_probs, 1) > 1) ? std(all_probs, dims=[1])[1, :] : zeros(T)

# Make the stddev bounds.
lower_p1 = min.(mean_probs .- 0, stddev_probs)
upper_p1 = min.(1 .- mean_probs, stddev_probs)
p5_unc, p6_unc = make_probability_plots(times[1:T], mean_probs[1:T]; include_gt=leader_idx, stddevs=(lower_p1, upper_p1))
plot!(p5_unc, title="", legend=:none)
plot!(p6_unc, title="", legend=:none)

filename = joinpath(topfolder_name, string("lf_uq_mc$(num_sims)_L$(leader_idx)_prob_P1.pdf"))
savefig(p5_unc, filename)

filename = joinpath(topfolder_name, string("lf_uq_mc$(num_sims)_L$(leader_idx)_prob_P2.pdf"))
savefig(p6_unc, filename)


