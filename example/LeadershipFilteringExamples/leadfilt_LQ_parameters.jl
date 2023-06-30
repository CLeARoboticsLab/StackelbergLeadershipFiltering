using StackelbergControlHypothesesFiltering

dt = 0.05
T = 201
t0 = 0.0
horizon = T * dt
# TODO(hamzah) - We do double the times as needed so that there's extra for the Stackelberg history. Make this tight.
times = dt * (cumsum(ones(2*T)) .- 1)

cont_lin_dyn = ShepherdAndSheepDynamics()
dyn = discretize(cont_lin_dyn, dt)
costs = ShepherdAndSheepCosts(dyn)
num_players = num_agents(dyn)

leader_idx = 2
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 0.; 1.; 0.; -1.; 0; 2; 0]
pos_unc = 1e-3
vel_unc = 1e-4
P₁ = Diagonal([pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc])

# Process noise uncertainty
Q = 1e-1 * Diagonal([1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4])


# TODO(hamzah) - vectorize this better
function generate_discrete_state_transition(p₁₁, p₂₂)

    distribs = [Bernoulli(p₁₁), Bernoulli(p₂₂)]

    # state transition matrix of state
    P = [ p₁₁  1-p₂₂;
         1-p₁₁  p₂₂]

    # The discrete state transition stays in state i with probability pᵢ.
    function discrete_state_transition(time_range, s_prev, s_probs, 𝒳_prev, s_actions, rng)

        @assert length(s_prev) == 1
        s_prev = s_prev[1]
        sample = rand(rng, distribs[s_prev], 1)

        # use markov chain to adjust over time
        other_state = (s_prev == 1) ? 2 : 1
        s_new = (isone(sample[1])) ? s_prev : other_state

        return [s_new]
    end
    return discrete_state_transition, P
end

# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 0.001 * I
zs = zeros(xdim(dyn), T)
Ts = 30
num_games = 1
num_particles = 50

p_transition = 0.98
p_init = 0.5


threshold = 1e-3
max_iters = 50
step_size = 1e-2


# Solve an LQ Stackelberg game based on the shepherd and sheep example.
Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, costs, T, leader_idx)
xs, us = unroll_feedback(dyn, times, Ps_strategies, x₁)

# Augment the remaining states so we have T+Ts-1 of them.
xs = hcat(xs, zeros(xdim(dyn), Ts-1))
us = [hcat(us[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

# Fill in z as noisy state measurements.
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(xs[:, tt], R))
end
