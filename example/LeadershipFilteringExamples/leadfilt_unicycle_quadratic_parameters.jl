using StackelbergControlHypothesesFiltering
using Random: seed!

seed!(0)


dt = 0.05
T = 201
t0 = 0.0
horizon = T * dt
# TODO(hamzah) - We do double the times as needed so that there's extra for the Stackelberg history. Make this tight.
times = dt * (cumsum(ones(2*T)) .- 1)

dyn = ShepherdAndSheepWithUnicycleDynamics(dt)
ss_costs = ShepherdAndSheepCosts(dyn; ctrl_const=.1)
num_players = num_agents(dyn)

# function make_quadratic_player_cost(si, ss_costs, player_idx)
#     f = get_as_function(ss_costs[player_idx])
#     return PlayerCost(f, si)
# end
# pc_cost_1 = make_quadratic_player_cost(dyn.sys_info, ss_costs, 1)
# pc_cost_2 = make_quadratic_player_cost(dyn.sys_info, ss_costs, 2)

# costs = [pc_cost_1, pc_cost_2]
costs = [QuadraticCostWithOffset(costs[1]), QuadraticCostWithOffset(costs[2])]
# costs = ss_costs

leader_idx = 1
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 1.; 0.; 0.; -1.; 2; 0; 0]
x₁ = [2.0, 1.0, 0.46364760900080615, 0.0, -2.0, -1.0000000000000004, -2.677945044588987, 0.0]

pos_unc = 1e-3
θ_inc = 1e-3
vel_unc = 1e-4
P₁ = Diagonal([pos_unc, pos_unc, θ_inc, vel_unc, pos_unc, pos_unc, θ_inc, vel_unc])

# Process noise uncertainty
Q = 5e-2 * Diagonal([1e-2, 1e-2, 1e-3, 1e-4, 1e-2, 1e-2, 1e-3, 1e-4])


# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = 0.01 * Matrix(I, xdim(dyn), xdim(dyn))
zs = zeros(xdim(dyn), T)
Ts = 30
num_games = 1
num_particles = 50

p_transition = 0.98
p_init = 0.5


threshold = 1e-3
max_iters = 50
step_size = 1e-2

# Generate the ground truth.

# leader_idx=2
gt_silq_num_runs=1

# config variables match the SILQGames configurations
gt_silq_threshold=2e-3
gt_silq_max_iters=1000
gt_silq_step_size=1e-2
gt_silq_verbose=true


# Set initial controls so that we can solve a Stackelberg game with SILQGames.
us_init = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
# for ii in 1:num_players

# angular velocities
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
