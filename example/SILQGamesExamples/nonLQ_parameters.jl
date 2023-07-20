using StackelbergControlHypothesesFiltering

using Random

seed = 1
rng = MersenneTwister(seed)

dt = 0.02
T = 251
horizon = T * dt
times = dt * (cumsum(ones(T)) .- 1)


dyn = ShepherdAndSheepWithUnicycleDynamics(dt)
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 1.; 7*pi/4; 0.; -1.; 2; -pi/4; 0] # unicycle dynamics

bound_val = 2.5
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
