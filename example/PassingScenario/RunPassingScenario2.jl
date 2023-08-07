using StackelbergControlHypothesesFiltering

using Dates
using LaTeXStrings
using LinearAlgebra: norm, Diagonal, I
using ProgressBars
using Random: MersenneTwister
using Distributions: Bernoulli, MvNormal

include("CreatePassingScenarioGame.jl")
include("GroundTruthUtils.jl")
include("PassingScenarioConfig.jl")
include("PassingScenarioPlotUtils.jl")

# Define game and timing related configuration.
num_players = 2

T = 151
t₀ = 0.0
dt = 0.05
horizon = T * dt
times = dt * cumsum(ones(2*T)) .- dt

# Get the configuration.
cfg = PassingScenarioConfig(collision_radius_m=0.2, lane_width_m=2.5)
                            # max_heading_deviation=pi/6)

# define limits for plots
limits = [-5., 120.]
limits_tuple = tuple(limits...)

# Defined the dynamics of the game.
dyn = create_passing_scenario_dynamics(num_players, dt)
si = dyn.sys_info

# Define the starting and goal state.
v_init = 10.
rlb_x = get_right_lane_boundary_x(cfg)
x₁ = [rlb_x/2; 10.; pi/2; v_init; rlb_x/2; 0.; pi/2; v_init]

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


# Generate a ground truth trajectory on which to run the leadership filter for a passing trajectory.
if T == 151
    x_refs, us_refs = get_passing_trajectory_scenario_151(cfg, x₁)
elseif T == 101
    x_refs, us_refs = get_passing_trajectory_scenario_101(cfg, x₁)
end
check_valid = get_validator(si, cfg)
@assert check_valid(x_refs, us_refs, times[1:T])






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
max_iters = 50
step_size = 1e-2

x̂s, P̂s, probs, pf, sg_objs = leadership_filter(dyn, costs, t₀, times,
                           T,         # simulation horizon
                           Ts,        # horizon over which the stackelberg game should be played,
                           num_games, # number of stackelberg games played for measurement
                           x₁,        # initial state at the beginning of simulation
                           P₁,        # initial covariance at the beginning of simulation
                           us_refs,      # the control inputs that the actor takes
                           zs,        # the measurements
                           R,
                           process_noise_distribution,
                           s_init_distrib,
                           discrete_state_transition;
                           threshold=threshold,
                           # check_valid=check_valid,
                           rng,
                           max_iters=max_iters,
                           step_size=step_size,
                           Ns=num_particles,
                           verbose=false)

using Dates
gr()

# Create the folder if it doesn't exist
folder_name = "passing_scenario_2_leadfilt_$(get_date_str())"
isdir(folder_name) || mkdir(folder_name)

# Generate the plots for the paper.
snapshot_freq = Int((T - 1)/10)
make_passing_scenario_pdf_plots(folder_name, snapshot_freq, cfg, limits, sg_objs[1].dyn, T, times, true_xs, true_us, probs, x̂s, zs, num_particles)

# This generates the gif.
filename = "passing_scenario_2.gif"
make_debug_gif(folder_name, filename, cfg, limits, dyn, T, times, true_xs, true_us, probs, x̂s, zs, Ts, num_particles, p_transition, num_games)

