using StackelbergControlHypothesesFiltering
using LaTeXStrings
using LinearAlgebra: norm, Diagonal, I
using Random: MersenneTwister
using Distributions: Bernoulli, MvNormal

include("CreatePassingScenarioGame.jl")
include("GroundTruthUtils.jl")
include("PassingScenarioConfig.jl")

# Define game and timing related configuration.
num_players = 2

T = 101
t₀ = 0.0
dt = 0.05
horizon = T * dt
times = dt * cumsum(ones(2*T)) .- dt

# Get the configuration.
cfg = PassingScenarioConfig(collision_radius_m=0.01)
                            # max_heading_deviation=pi/6)

# Defined the dynamics of the game.
dyn = create_passing_scenario_dynamics(num_players, dt)
si = dyn.sys_info

# Define the starting and goal state.
v_init = 10.
rlb_x = get_right_lane_boundary_x(cfg)
x₁ = [rlb_x/2; 10.; pi/2; v_init; rlb_x/1.5; 0.; pi/2; v_init]

p1_goal = vcat([x₁[1]; 60; pi/2; v_init], zeros(4))
p2_goal = vcat(zeros(4),                [x₁[5]; 50.; pi/2; v_init])

# Define the costs for the agents.
num_subcosts = 14
weights_p1 = ones(num_subcosts)
weights_p2 = ones(num_subcosts)

# Adjust goal tracking weights.
weights_p1[1] = 1.
weights_p2[1] = 1.
# weights_p1[2] = 1//2
costs = create_passing_scenario_costs(cfg, si, weights_p1, weights_p2, p1_goal, p2_goal)


# Generate a ground truth trajectory on which to run the leadership filter.
gt_threshold=1e-3
gt_max_iters=120
gt_step_size=1e-2
gt_verbose=true
gt_num_runs=1
sg_obj = initialize_silq_games_object(gt_num_runs, T, dyn, costs;
                                      threshold=gt_threshold, max_iters=gt_max_iters, step_size=gt_step_size, verbose=gt_verbose)

# An initial control estimate.
gt_leader_idx = 1
us_1 = [zeros(udim(si, 1), T), zeros(udim(si, 1), T)]
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = generate_gt_from_silqgames(sg_obj, gt_leader_idx, times, x₁, us_1)
plot_silqgames_gt(dyn, times[1:T], xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs)


# Run the leadership filter.

# Initial condition chosen randomly. Ensure both have relatively low speed.
pos_unc = 1e-2
θ_inc = 1e-2
vel_unc = 1e-3
P₁ = Diagonal([pos_unc, pos_unc, θ_inc, vel_unc, pos_unc, pos_unc, θ_inc, vel_unc])

# Process noise uncertainty
Q = 1e-1 * Diagonal([1e-2, 1e-2, 1e-3, 1e-4, 1e-2, 1e-2, 1e-3, 1e-4])


# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 0.01 * I
zs = zeros(xdim(dyn), T)
Ts = 20
num_games = 1
num_particles = 100

p_transition = 0.98
p_init = 0.5

discrete_state_transition, state_trans_P = generate_discrete_state_transition(p_transition, p_transition)
s_init_distrib = Bernoulli(p_init)

process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q)


threshold = 1e-2
max_iters = 50
step_size = 1e-2

# Augment the remaining states so we have T+Ts-1 of them.
true_xs = hcat(xs_k, zeros(xdim(dyn), Ts-1))
true_us = [hcat(us_k[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

# Fill in z as noisy state measurements.
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(true_xs[:, tt], R))
end

x̂s, P̂s, probs, pf, sg_objs = leadership_filter(dyn, costs, t₀, times,
                           T,         # simulation horizon
                           Ts,        # horizon over which the stackelberg game should be played,
                           num_games, # number of stackelberg games played for measurement
                           x₁,        # initial state at the beginning of simulation
                           P₁,        # initial covariance at the beginning of simulation
                           us_k,      # the control inputs that the actor takes
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

using Dates
using Plots
using Printf
using ProgressBars
gr()

rotated_true_xs = rotate_state(dyn, true_xs)
rotated_zs = rotate_state(dyn, zs)
rotated_x̂s = rotate_state(dyn, x̂s)
rotate_particle_state(xs) = rotate_state(dyn, xs)

# This generates a pdf.

# Create the folder if it doesn't exist
folder_name = "passing_scenario_1_leadfilt_$(Dates.now())"
isdir(folder_name) || mkdir(folder_name)

snapshot_freq = Int((T - 1)/10)
iter1 = ProgressBar(2:snapshot_freq:T)
ii = 1

# Only needs to be generated once.
p1a = plot_leadership_filter_positions(sg_objs[1].dyn, rotated_true_xs[:, 1:T], rotated_x̂s[:, 1:T], rotated_zs[:, 1:T])
plot!(p1a, legend=:outertopright, ylabel=L"$-x$ (m)", xlabel=L"$y$ (m)", ylimit=(-(cfg.lane_width_m+1), cfg.lane_width_m+1), xlimit=(-5., 75.))

pos_main_filepath = joinpath(folder_name, "lf_nonlq_positions_main_L$(leader_idx).pdf")
savefig(p1a, pos_main_filepath)

for t in iter1
    p1b = plot_leadership_filter_measurement_details(num_particles, sg_objs[t], rotated_true_xs[:, 1:T], rotated_x̂s; transform_particle_fn=rotate_particle_state)
    plot!(p1b, legend=:outertopright, ylabel=L"$-x$ (m)", xlabel=L"$y$ (m)", ylimit=(-(cfg.lane_width_m+1), cfg.lane_width_m+1), xlimit=(-5., 75.))

    p5, p6 = make_probability_plots(leader_idx, times[1:T], t, probs[1:T]; include_gt=false)
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
p1a = plot_leadership_filter_positions(dyn, rotated_true_xs[:, 1:T], rotated_x̂s[:, 1:T], rotated_zs[:, 1:T])
plot!(p1a, title=title, legend=:outertopright, ylabel=L"$-x$ (m)", xlabel=L"$y$ (m)", ylimit=(-(cfg.lane_width_m+1), cfg.lane_width_m+1), xlimit=(-5., 75.))

iter = ProgressBar(2:T)
anim = @animate for t in iter
    p = @layout [a b; grid(1, 3); e f]

    plot_title = string("LF (", t, "/", T, ") on Stack(L=P", leader_idx, "), Ts=", Ts, ", Ns=", num_particles, ", p(transition)=", p_transition, ", #games: ", num_games)
    p1b = plot_leadership_filter_measurement_details(num_particles, sg_objs[t], rotated_true_xs[:, 1:T], rotated_x̂s[:, 1:T]; transform_particle_fn=rotate_particle_state)
    plot!(p1b, legend=:outertopright, ylabel=L"$-x$ (m)", xlabel=L"$y$ (m)", ylimit=(-(cfg.lane_width_m+1), cfg.lane_width_m+1), xlimit=(-5., 75.))

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
    plot!(p4, times[1:T], true_us[1][1, 1:T], label="P1 ω")
    plot!(p4, times[1:T], true_us[2][1, 1:T], label="P2 ω")
    plot!(p4, times[1:T], true_us[1][2, 1:T], label="P1 a")
    plot!(p4, times[1:T], true_us[2][2, 1:T], label="P2 a")
    plot!(p4, [times[t], times[t]], [-1, 1], label="", color=:black)

    # probability plots 5 and 6
    title5 = "Probability over time for P1"
    title6 = "Probability over time for P2"
    p5, p6 = make_probability_plots(leader_idx, times[1:T], t, probs[1:T]; include_gt=false)
    plot!(p5, title=title5)
    plot!(p6, title=title6)

    plot(p1a, p1b, p2, p3, p4, p5, p6, plot_title=plot_title, layout = p, size=(1260, 1080))
end


# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
filename = joinpath(folder_name, "passing_scenario_1.gif")
gif(anim, filename, fps=10)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
