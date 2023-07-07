using StackelbergControlHypothesesFiltering

dt = 0.01
T = 401
t0 = 0.0
horizon = T * dt
# TODO(hamzah) - We do double the times as needed so that there's extra for the Stackelberg history. Make this tight.
times = dt * (cumsum(ones(2*T)) .- 1)

dyn = ShepherdAndSheepWithUnicycleDynamics(dt)
bound_val = 2.1
use_autodiff = true
costs = ShepherdAndSheepWithLogBarrierOverallCosts(dyn, bound_val, use_autodiff)
# costs = ShepherdAndSheepWithLogBarrierOverallCosts(dyn, (-bound_val, bound_val), (0., bound_val))
num_players = num_agents(dyn)

leader_idx = 1
# Initial condition chosen randomly. Ensure both have relatively low speed.
# top half of plane
x₁ = [-2.; 1.; 0; 0.; 1.; -2; 0; 0]

# opposite diagonals
# x₁ = [2.; 0.; -1.; 0.; -1.; 0; 2; 0]
# x₁ = [1.; 0.; 0.01; 0.; -1.; 0; -0.01; 0]
# x₁ = rand(rng, 8)
# x₁[[2, 4, 6, 8]] .= 0

pos_unc = 1e-3
vel_unc = 1e-4
P₁ = Diagonal([pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc])

# Process noise uncertainty
Q = 1e-2 * Diagonal([1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4])


# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 0.001 * I
zs = zeros(xdim(dyn), T)
Ts = 20
num_games = 1
num_particles = 100

p_transition = 0.98
p_init = 0.7

threshold = 1e-3
max_iters = 50
step_size = 2e-2

gt_silq_num_runs=1

# config variables
gt_silq_threshold=1e-3
gt_silq_max_iters=1000
gt_silq_step_size=1e-2
gt_silq_verbose=true


# Set initial controls so that we can solve a Stackelberg game with SILQGames.
us_init = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]

# # angular velocities
# us_init[1][1,:] .= -.03
# us_init[2][1,:] .= -.01

# # accelerations
# us_init[1][2,:] .= -.3
# us_init[2][2,:] .= .3


# Generate the ground truth.
sg_obj = initialize_silq_games_object(gt_silq_num_runs, T, dyn, costs;
                                      threshold=gt_silq_threshold, max_iters=gt_silq_max_iters, step_size=gt_silq_step_size, verbose=gt_silq_verbose)
true_xs, true_us, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times[1:T], x₁, us_init)

# Augment the remaining states so we have T+Ts-1 of them.
xs = hcat(true_xs, zeros(xdim(dyn), Ts-1))
us = [hcat(true_us[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

# Fill in z as noisy state measurements.
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(true_xs[:, tt], R))
end
