using StackelbergControlHypothesesFiltering

dt = 0.05
T = 101
horizon = T * dt
times = dt * (cumsum(ones(T)) .- 1)

dyn = ShepherdAndSheepDynamics(dt)
costs = ShepherdAndSheepCosts()
num_players = num_agents(dyn)

leader_idx = 1
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 0.; 1.; 0.; -1.; 0; 2; 0]
# x₁ = [1.; 0.; 0.01; 0.; -1.; 0; -0.01; 0]
# x₁ = ones()
# x₁[[2, 4, 6, 8]] .= 0

# Initial controls
us_1 = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
for ii in 1:num_players
    us_1[ii][1,:] .= -1.
    us_1[ii][2,:] .= -1.1
end
# duration = (T-1) * dt
# us_1[ii][1, :] .= (xf[3] - x0[3]) / duration # omega
# us_1[ii][2, :] .= (xf[4] - x0[4]) / duration # accel

# Test with ideal solution.
# Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, costs, T, leader_idx)
# _, us_1 = unroll_feedback(dyn, times, Ps_strategies, x₁)