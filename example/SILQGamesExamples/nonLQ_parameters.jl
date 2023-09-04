using StackelbergControlHypothesesFiltering

using Random

seed = 1
rng = MersenneTwister(seed)

dt = 0.05
T = 201
horizon = T * dt
times = dt * (cumsum(ones(T)) .- 1)


dyn = ShepherdAndSheepWithUnicycleDynamics(dt)
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 1.; 7*pi/4; 0.; -1.; 2; -pi/4; 0] # unicycle dynamics
x₁ = [2.0, 1.0, -2.677945044588987, 0.0, -0.7533842212760272, 2.105329478996652, -1.2271487177940905, 0.0]

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
