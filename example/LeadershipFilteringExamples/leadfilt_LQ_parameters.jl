using StackelbergControlHypothesesFiltering

dt = 0.05
T = 101
t0 = 0.0
horizon = T * dt
# TODO(hamzah) - We do double the times as needed so that there's extra for the Stackelberg history. Make this tight.
times = dt * (cumsum(ones(2*T)) .- 1)

dyn = ShepherdAndSheepDynamics(dt)
costs = ShepherdAndSheepCosts()
num_players = num_agents(dyn)

leader_idx = 2
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 0.; 1.; 0.; -1.; 0; 2; 0]
pos_unc = 1e-2
vel_unc = 1e-3
P₁ = Diagonal([pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc])

# x₁ = [1.; 0.; 0.01; 0.; -1.; 0; -0.01; 0]
# x₁ = rand(8)
# x₁[[2, 4, 6, 8]] .= 0

# Initial controls
# us_1 = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
# for ii in 1:num_players
#     us_1[ii][1,:] .= -1.
#     us_1[ii][2,:] .= -.1
# end
# duration = (T-1) * dt
# us_1[ii][1, :] .= (xf[3] - x0[3]) / duration # omega
# us_1[ii][2, :] .= (xf[4] - x0[4]) / duration # accel

# Test with ideal solution.
# Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, costs, T, leader_idx)
# _, us_1 = unroll_feedback(dyn, times, Ps_strategies, x₁)
