# This file sets up a simple scenario and runs it on a Stackelberg game (with or without noise).
using StackelbergControlHypothesesFiltering

using Distributions
using LaTeXStrings
using LinearAlgebra
using Random
using Plots

gr()

include("leadfilt_LQ_parameters.jl")


discrete_state_transition, state_trans_P = generate_discrete_state_transition(p_transition, p_transition)
s_init_distrib = Bernoulli(p_init)

process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q)


x̂s, P̂s, probs, pf, sg_objs = leadership_filter(dyn, costs, t0, times,
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
                           verbose=false,
                           ensure_pd=false)

true_xs = xs

using Dates
using Plots
using Printf
using ProgressBars
gr()


# This generates a pdf.


# Create the folder if it doesn't exist
folder_name = "lq_L$(leader_idx)_leadfilt_$(Dates.now())"
isdir(folder_name) || mkdir(folder_name)

snapshot_freq = Int((T - 1)/10)
iter1 = ProgressBar(2:snapshot_freq:T)
ii = 1
for t in iter1
    p1 = plot_leadership_filter_position(num_particles, sg_objs[t], true_xs[:, 1:T], x̂s[:, 1:T], zs[:, 1:T])
    plot!(p1, legend=:bottomleft)

    p5, p6 = make_probability_plots(times[1:T], t, probs)
    plot!(p5, title="")
    plot!(p6, title="")

    pos_filepath = joinpath(folder_name, "0$(ii)_lf_$(t)_lq_positions_L$(leader_idx).pdf")
    prob1_filepath = joinpath(folder_name, "0$(ii)_lf_$(t)_lq_probs_P1_L$(leader_idx).pdf")
    prob2_filepath = joinpath(folder_name, "0$(ii)_lf_$(t)_lq_probs_P2_L$(leader_idx).pdf")
    savefig(p1, pos_filepath)
    savefig(p5, prob1_filepath)
    savefig(p6, prob2_filepath)

    global ii += 1
end


# This generates the gif.
iter = ProgressBar(2:T)
anim = @animate for t in iter
    p = @layout [a b; c d; e f]

    plot_title = string("LF (", t, "/", T, ") on Stack(L=P", leader_idx, "), Ts=", Ts, ", Ns=", num_particles, ", p(transition)=", p_transition, ", #games: ", num_games)
    title="x-y plot of agent positions over time"
    p1 = plot_leadership_filter_position(num_particles, sg_objs[t], true_xs, zs)
    plot!(p1, title=title, legend=:outertopright)

    _, p_px, p_py, p_vx, p_vy, p_ax, p_ay = plot_states_and_controls(dyn, times[1:T], x̂s, us)

    # plot 2 - positions
    title1 = "LF estimated pos. (x̂/ŷ)"
    plot!(p_px, [times[t], times[t]], [-2, 2], label="", color=:black)
    plot!(p_py, [times[t], times[t]], [-2, 2], label="", color=:black)
    p2 = plot!(p_px, p_py, overlay = true, title=title1)

    # plot 3 - velocities
    title2 = "LF estimated velocity (v̂) over time"
    plot!(p_vx, [times[t], times[t]], [-2, 2], label="", color=:black)
    plot!(p_vy, [times[t], times[t]], [-2, 2], label="", color=:black)
    p3 = plot!(p_vx, p_vy, overlay = true, title=title2)

    # plot 4 - acceleration inputs
    title4 = "Input acceleration controls (u) over time"
    p4 = plot(legend=:outertopright, xlabel="t (s)", ylabel="accel. (m/s^2)", title=title4)
    plot!(p4, times[1:T], us[1][1, 1:T], label="P1 ax")
    plot!(p4, times[1:T], us[1][2, 1:T], label="P1 ay")
    plot!(p4, times[1:T], us[2][1, 1:T], label="P2 ax")
    plot!(p4, times[1:T], us[2][2, 1:T], label="P2 ay")
    plot!(p4, [times[t], times[t]], [-2, 2], label="", color=:black)

    # probability plots 5 and 6
    title5 = "Probability over time for P1"
    title6 = "Probability over time for P2"
    p5, p6 = make_probability_plots(times[1:T], t, probs)
    plot!(p5, title=title5)
    plot!(p6, title=title6)

    plot(p1, p2, p3, p4, p5, p6, plot_title=plot_title, layout = p, size=(1260, 1080))
end

# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
filename = joinpath(folder_name, "lq_leadfilt.gif")
gif(anim, filename, fps=10)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
