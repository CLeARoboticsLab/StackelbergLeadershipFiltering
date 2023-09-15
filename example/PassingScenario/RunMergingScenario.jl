using StackelbergControlHypothesesFiltering

using Dates
using LaTeXStrings
using LinearAlgebra: norm, Diagonal, I
using Plots
using ProgressBars
using Random: MersenneTwister
using Distributions: Bernoulli, MvNormal
#gr()

include("CreateMergingScenarioGame.jl")
include("GroundTruthUtils.jl")
include("MergingScenarioConfig.jl")
include("PassingScenarioPlotUtils.jl")

# Define game and timing related configuration.
num_players = 2

T = 101
t₀ = 0.0
dt = 0.05
horizon = T * dt
times = dt * cumsum(ones(2*T)) .- dt

# Get the configuration.
cfg = MergingScenarioConfig(collision_radius_m=0.2, lane_width_m=2.5, max_heading_deviation=pi/6)

# define limits for plots
limits = [-5., 120.]
limits_tuple = tuple(limits...)

# Defined the dynamics of the game.
dyn = create_merging_scenario_dynamics(num_players, dt)
si = dyn.sys_info

# Define the starting and goal state.
v_goal = 10.
lw_m = cfg.lane_width_m
# x₁ = [-lw_m/2+0.25; 20.; pi/2 - 0.001*pi; 2*v_init; lw_m/2; 0.; pi/2+0.01; v_init]
# x₁ = [-lw_m/2; 20.; pi/2 - 0.001*pi; v_init; lw_m/2; 0.; pi/2+0.01; v_init]
# x₁ = [-lw_m/2; 15.; pi/2; v_goal; lw_m/2; 0.; pi/2; v_goal]

# Generate a ground truth trajectory on which to run the leadership filter for a merging trajectory.
us_refs, x₁, p1_goal, p2_goal = get_merging_trajectory_p1_first_101(cfg)
us_refs, x₁, p1_goal, p2_goal = get_merging_trajectory_p2_reverse_101(cfg)
us_refs, x₁, p1_goal, p2_goal = get_merging_trajectory_p2_flipped_101(cfg)

p1_on_left = (x₁[1] < 0 && x₁[5] > 0)
@assert xor(x₁[1] < 0 && x₁[5] > 0, x₁[1] > 0 && x₁[5] < 0)

println(p1_on_left)

# us_refs, x₁ = get_merging_trajectory_p1_same_start_101(cfg)
# us_refs = [zeros(2, T) for ii in 1:2]

# Define the costs for the agents.
num_subcosts = 13
weights_p1 = ones(num_subcosts)
weights_p2 = ones(num_subcosts)

# Adjust goal tracking weights.
weights_p1[2] = 1.
weights_p2[2] = 1.
costs = create_merging_scenario_costs(cfg, si, weights_p1, weights_p2, p1_goal, p2_goal; p1_on_left)


x_refs = unroll_raw_controls(dyn, times[1:T], us_refs, x₁)
check_valid = get_validator(si, cfg)
@assert check_valid(x_refs, us_refs, times[1:T]; p1_on_left)
plot_silqgames_gt(dyn, cfg, times[1:T], x_refs, us_refs)

# gt_threshold = 1e-2
# gt_max_iters = 1000
# gt_step_size = 1e-2
# gt_verbose = true
# gt_leader = 1
# sg = initialize_silq_games_object(1, T, dyn, costs;
#                                       # state_reg_param=1e-2, control_reg_param=1e-2, ensure_pd=true,
#                                       threshold = gt_threshold, max_iters = gt_max_iters, step_size=gt_step_size, verbose=gt_verbose)
#                                       # ss_reduce=1e-2, α_min=1e-2, max_linesearch_iters=10,
#                                       # check_valid=(xs, us, ts)->true, verbose=false, ignore_Kks=true, ignore_xkuk_iters=true)
# xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = generate_gt_from_silqgames(sg, gt_leader, times, x₁, us_refs)
# x_refs, us_refs = xs_k, us_k

# S = 101
# p_ts = times[1:S]
# p_xs = x_refs[:, 1:S]
# p_us = [us_refs[ii][:, 1:S] for ii in 1:2]
# check_valid(p_xs, p_us, p_ts; p1_on_left)
# plot_silqgames_gt(dyn, cfg, p_ts, p_xs, p_us)



# Run the leadership filter.

# Initial condition chosen randomly. Ensure both have relatively low speed.
pos_unc = 1e-3
θ_inc = 1e-4
vel_unc = 1e-3
P₁ = Diagonal([pos_unc, pos_unc, θ_inc, vel_unc, pos_unc, pos_unc, θ_inc, vel_unc])

# Process noise uncertainty
Q = 1e-2 * Diagonal([1e-2, 1e-2, 1e-3, 1e-2, 1e-2, 1e-2, 1e-3, 1e-2])


# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 1e-2 * I
zs = zeros(xdim(dyn), T)
Ts = 20
num_games = 1
num_particles = 100

p_transition = 0.98
p_init = 0.5

discrete_state_transition, state_trans_P = generate_discrete_state_transition(p_transition, p_transition)
s_init_distrib = Bernoulli(p_init)

process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q)


# x_refs = xs_k
# us_refs = us_k

# Augment the remaining states so we have T+Ts-1 of them.
true_xs = hcat(x_refs, zeros(xdim(dyn), Ts-1))
true_us = [hcat(us_refs[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

# Fill in z as noisy state measurements.
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(true_xs[:, tt], R))
end

# l = @layout [a{0.3h}; grid(2, 3)]

# pos_plot, p2, p3, p4, p5, p6, p7 = plot_states_and_controls(dyn, times[1:T], rotate_state(dyn, x_refs[:, 1:T]), us_refs)

# plot_zs = rotate_state(dyn, zs)
# scatter!(pos_plot, plot_zs[1, :], plot_zs[2, :], color=:turquoise, label="", ms=0.5)
# scatter!(pos_plot, plot_zs[5, :], plot_zs[6, :], color=:orange, label="", ms=0.5)

# plot(pos_plot, p2, p3, p4, p5, p6, p7, layout=l)


threshold = 1e-2
max_iters = 100
step_size = 1e-2

x̂s, P̂s, probs, pf, sg_objs, iter_timings = leadership_filter(dyn, costs, t₀, times,
                           T,         # simulation horizon
                           Ts,        # horizon over which the stackelberg game should be played,
                           num_games, # number of stackelberg games played for measurement
                           x₁,        # initial state at the beginning of simulation
                           P₁,        # initial covariance at the beginning of simulation
                           us_refs,   # the control inputs that the actor takes
                           zs,        # the measurements
                           1.1 * R,
                           process_noise_distribution,
                           s_init_distrib,
                           discrete_state_transition;
                           threshold=threshold,
                           # check_valid=check_valid,
                           rng,
                           max_iters=max_iters,
                           step_size=step_size,
                           Ns=num_particles,
                           verbose=true)

using Dates
gr()

# Create the folder if it doesn't exist
folder_name = "merging_scenario_2_leadfilt_$(get_date_str())"
isdir(folder_name) || mkdir(folder_name)

# Generate the plots for the paper.
snapshot_freq = Int((T - 1)/10)
make_merging_scenario_pdf_plots(folder_name, snapshot_freq, cfg, limits, sg_objs[1].dyn, T, times, true_xs, true_us, probs, x̂s, zs, num_particles)
# make_driving_scenario_pdf_plots(folder_name, snapshot_freq, cfg, limits, dyn, horizon, times, true_xs, true_us, probs, x̂s, zs, num_particles)

# This generates the gif.
filename = "merging_scenario_2.gif"
make_debug_gif(folder_name, filename, cfg, limits, dyn, T, times, true_xs, true_us, probs, x̂s, zs, Ts, num_particles, p_transition, num_games)

