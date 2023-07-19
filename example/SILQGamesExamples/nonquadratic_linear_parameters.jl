using StackelbergControlHypothesesFiltering

using Random

seed = 1
rng = MersenneTwister(seed)

dt = 0.02
T = 251
horizon = T * dt
times = dt * (cumsum(ones(T)) .- 1)

dyn = ShepherdAndSheepDynamics(); dyn = discretize(dyn, dt)
# top half of plane
x₁ = [2.; 0.; 1.; 0.; -1.; 0; 2; 0] # double integrator dynamics
x₁[[2, 4, 6, 8]] .= 0

# opposite diagonals
# x₁ = [2.; 0.; -1.; 0.; -1.; 0; 2; 0]
# x₁ = [1.; 0.; 0.01; 0.; -1.; 0; -0.01; 0]
# x₁ = rand(rng, 8)


# dyn = ShepherdAndSheepWithUnicycleDynamics(dt)
# # Initial condition chosen randomly. Ensure both have relatively low speed.
# x₁ = [2.; 1.; -3*pi/4; 0.; -1.; 2; -pi/4; 0] # unicycle dynamics

bound_val = 2.1
use_autodiff = true
costs = ShepherdAndSheepWithLogBarrierOverallCosts(dyn, (-bound_val, bound_val), (-bound_val, bound_val), use_autodiff)
num_players = num_agents(dyn)

leader_idx = 2


# Initial controls
us_1 = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
# for ii in 1:num_players
#     us_1[ii][1,:] .= -1.
#     us_1[ii][2,:] .= -.1
# end

# Test with ideal solution, with noise.
# Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, costs, T, leader_idx)
# _, us_1 = unroll_feedback(dyn, times, Ps_strategies, x₁)
# us_1[1] += 0.1 * rand(rng, udim(dyn, 1), T)
# us_1[2] += 0.1 * rand(rng, udim(dyn, 1), T)
