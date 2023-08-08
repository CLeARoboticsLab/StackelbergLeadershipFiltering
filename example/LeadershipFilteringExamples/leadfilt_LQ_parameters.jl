using StackelbergControlHypothesesFiltering

dt = 0.05
T = 201
t0 = 0.0
horizon = T * dt
# TODO(hamzah) - We do double the times as needed so that there's extra for the Stackelberg history. Make this tight.
times = dt * (cumsum(ones(2*T)) .- 1)

cont_lin_dyn = ShepherdAndSheepDynamics()
dyn = discretize(cont_lin_dyn, dt)
ss_costs = ShepherdAndSheepCosts(dyn)
num_players = num_agents(dyn)


fs = get_as_function.(ss_costs)
pc_cost_1 = PlayerCost(fs[1], dyn.sys_info)
pc_cost_2 = PlayerCost(fs[2], dyn.sys_info)


leader_idx = 1
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 0.; 1.; 0.; -1.; 0; 2; 0]
pos_unc = 1e-3
vel_unc = 1e-4
P₁ = Diagonal([pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc])

# Process noise uncertainty
Q = 1e-2 * Diagonal([1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4])


# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = 0.005 * Matrix(I, xdim(dyn), xdim(dyn))
zs = zeros(xdim(dyn), T)
Ts = 30
num_games = 1
num_particles = 50

p_transition = 0.98
p_init = 0.5


# We use these in the measurement model.
threshold = 1.5e-2
max_iters = 50
step_size = 1e-2


# Solve an LQ Stackelberg game based on the shepherd and sheep example.
Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, ss_costs, T, leader_idx)
xs, us = unroll_feedback(dyn, times, Ps_strategies, x₁)

# Augment the remaining states so we have T+Ts-1 of them.
true_xs = hcat(xs, zeros(xdim(dyn), Ts-1))
true_us = [hcat(us[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

# Fill in z as noisy state measurements.
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(xs[:, tt], R))
end
