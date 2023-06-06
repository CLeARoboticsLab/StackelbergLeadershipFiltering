using StackelbergControlHypothesesFiltering

using Random

seed = 1
rng = MersenneTwister(seed)

dt = 0.05
T = 301
horizon = T * dt
times = dt * (cumsum(ones(T)) .- 1)

dyn = ShepherdAndSheepWithUnicycleDynamics(dt)
costs = UnicycleShepherdAndSheepWithQuadraticCosts()
num_players = num_agents(dyn)

leader_idx = 1
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 1.; 0.; 0.; -1.; 2; 0; 0]
# x₁ = [1.; 0.; 0.01; 0.; -1.; 0; -0.01; 0]
# x₁ = rand(rng, 8)
x₁[[3, 4, 7, 8]] .= 0

# Initial controls
us_1 = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
# for ii in 1:num_players

# angular velocities
us_1[1][1,:] .= -.03
us_1[2][1,:] .= -.01

# accelerations
us_1[1][2,:] .= -.3
us_1[2][2,:] .= .3

# end
# duration = (T-1) * dt
# us_1[ii][1, :] .= (xf[3] - x0[3]) / duration # omega
# us_1[ii][2, :] .= (xf[4] - x0[4]) / duration # accel

# Test with ideal solution, with noise.
# Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, costs, T, leader_idx)
# _, us_1 = unroll_feedback(dyn, times, Ps_strategies, x₁)
# us_1[1] += 0.1 * rand(rng, udim(dyn, 1), T)
# us_1[2] += 0.1 * rand(rng, udim(dyn, 1), T)
