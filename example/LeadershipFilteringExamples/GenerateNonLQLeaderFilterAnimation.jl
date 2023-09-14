# This file sets up a simple scenario and runs the leadership filter on a Stackelberg game with unicycle dynamics and quadratic cost.
using StackelbergControlHypothesesFiltering

using Distributions
using LinearAlgebra
using Random
using Plots

gr()

include("leadfilt_non_LQ_parameters.jl")


discrete_state_transition, state_trans_P = generate_discrete_state_transition(p_transition, p_transition)
s_init_distrib = Bernoulli(p_init)

process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q)


x̂s, P̂s, probs, pf, sg_objs, iter_timings = leadership_filter(dyn, costs, t0, times,
                           T,         # simulation horizon
                           Ts,        # horizon over which the stackelberg game should be played,
                           num_games, # number of stackelberg games played for measurement
                           x₁,        # initial state at the beginning of simulation
                           P₁,        # initial covariance at the beginning of simulation
                           us,        # the control inputs that the actor takes
                           zs,        # the measurements
                           R,
                           process_noise_distribution,
                           s_init_distrib,
                           discrete_state_transition;
                           threshold=threshold,
                           rng,
                           max_iters=max_iters,
                           step_size=step_size,
                           Ns=num_particles,
                           verbose=false)

true_xs = xs

using Dates
using Plots
using Printf
using ProgressBars
gr()

# This generates a pdf.

# Create the folder if it doesn't exist
folder_name = "nonlq_L$(leader_idx)_leadfilt_$(get_date_str())"
isdir(folder_name) || mkdir(folder_name)

snapshot_freq = Int((T - 1)/10)
iter1 = ProgressBar(2:snapshot_freq:T)
ii = 1

# Only needs to be generated once.
p1a = plot_leadership_filter_positions(sg_objs[1].dyn, true_xs[:, 1:T], x̂s[:, 1:T])
p1_meas = plot_leadership_filter_measurements(sg_objs[1].dyn, true_xs[:, 1:T], zs[:, 1:T])

pos_main_filepath = joinpath(folder_name, "lf_nonlq_positions_main_L$(leader_idx).pdf")
savefig(p1a, pos_main_filepath)

pos_meas_filepath = joinpath(folder_name, "lf_nonlq_positions_meas_L$(leader_idx).pdf")
savefig(p1_meas, pos_meas_filepath)

for t in iter1
    
    plot!(p1a, legend=:bottomleft)

    p1b = plot_leadership_filter_measurement_details(num_particles, sg_objs[t], true_xs[:, 1:T], x̂s)

    p5 = make_probability_plots(times[1:T], probs[1:T]; t_idx=t, include_gt=leader_idx, player_to_plot=1)
    p6 = make_probability_plots(times[1:T], probs[1:T]; t_idx=t, include_gt=leader_idx, player_to_plot=2)
    plot!(p5, title="")
    plot!(p6, title="")

    pos2_filepath = joinpath(folder_name, "0$(ii)_lf_t$(t)_nonlq_positions_detail_L$(leader_idx).pdf")
    prob1_filepath = joinpath(folder_name, "0$(ii)_lf_t$(t)_nonlq_probs_P1_L$(leader_idx).pdf")
    prob2_filepath = joinpath(folder_name, "0$(ii)_lf_t$(t)_nonlq_probs_P2_L$(leader_idx).pdf")
    
    savefig(p1b, pos2_filepath)
    savefig(p5, prob1_filepath)
    savefig(p6, prob2_filepath)

    global ii += 1
end


# This generates the gif.

# This plot need not be in the loop.
title="x-y plot of agent positions over time"
p1a = plot_leadership_filter_positions(sg_objs[1].dyn, true_xs[:, 1:T], x̂s[:, 1:T])
plot!(p1a, title=title, legend=:outertopright)

iter = ProgressBar(2:T)
anim = @animate for t in iter
    p = @layout [a b; grid(1, 3); e f]

    plot_title = string("LF (", t, "/", T, ") on Stack(L=P", leader_idx, "), Ts=", Ts, ", Ns=", num_particles, ", p(transition)=", p_transition, ", #games: ", num_games)
    p1b = plot_leadership_filter_measurement_details(num_particles, sg_objs[t], true_xs[:, 1:T], x̂s[:, 1:T])

    _, p_px, p_py, p_θ, p_v, _, _ = plot_states_and_controls(dyn, times[1:T], true_xs[:, 1:T], us)

    # plot 2 - positions
    title1 = "LF est. pos. (x̂/ŷ)"
    plot!(p_px, [times[t], times[t]], [-2, 2], label="", color=:black)
    plot!(p_py, [times[t], times[t]], [-2, 2], label="", color=:black)
    p2 = plot!(p_px, p_py, overlay = true, title=title1)

    # plot 3 - velocities
    title2 = "LF est. heading/velocity (θ̂/v̂)"
    plot!(p_θ, [times[t], times[t]], [-2, 2], label="", color=:black)
    plot!(p_v, [times[t], times[t]], [-2, 2], label="", color=:black)
    p3 = plot!(p_θ, p_v, overlay = true, title=title2)

    # plot 4 - accel. controls
    title5 = "Input acceleration controls (u) over time"
    p4 = plot(legend=:outertopright, xlabel="t (s)", ylabel="accel. (m/s^2)", title=title5)
    plot!(p4, times[1:T], us[1][1, 1:T], label=L"\mathcal{A}_1 ω")
    plot!(p4, times[1:T], us[2][1, 1:T], label=L"\mathcal{A}_2 ω")
    plot!(p4, times[1:T], us[1][2, 1:T], label=L"\mathcal{A}_1 a")
    plot!(p4, times[1:T], us[2][2, 1:T], label=L"\mathcal{A}_2 a")
    plot!(p4, [times[t], times[t]], [-1, 1], label="", color=:black)

    # probability plots 5 and 6
    title5 = "Probability over time for P1"
    title6 = "Probability over time for P2"
    p5 = make_probability_plots(times[1:T], probs[1:T]; t_idx=t, include_gt=leader_idx, player_to_plot=1)
    p6 = make_probability_plots(times[1:T], probs[1:T]; t_idx=t, include_gt=leader_idx, player_to_plot=2)
    plot!(p5, title=title5)
    plot!(p6, title=title6)

    plot(p1a, p1b, p2, p3, p4, p5, p6, plot_title=plot_title, layout = p, size=(1260, 1080))
end

# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
filename = joinpath(folder_name, "non_lq_leadfilt.gif")
gif(anim, filename, fps=10)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
