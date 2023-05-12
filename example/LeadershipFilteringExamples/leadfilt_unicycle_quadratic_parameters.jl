using StackelbergControlHypothesesFiltering

dt = 0.05
T = 301
t0 = 0.0
horizon = T * dt
# TODO(hamzah) - We do double the times as needed so that there's extra for the Stackelberg history. Make this tight.
times = dt * (cumsum(ones(2*T)) .- 1)

dyn = ShepherdAndSheepWithUnicycleDynamics()
costs = UnicycleShepherdAndSheepWithQuadraticCosts()
num_players = num_agents(dyn)

leader_idx = 1
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 1.; 0.; 0.; -1.; 2; 0; 0]
pos_unc = 1e-3
θ_inc = 1e-3
vel_unc = 1e-4
P₁ = Diagonal([pos_unc, pos_unc, θ_inc, vel_unc, pos_unc, pos_unc, θ_inc, vel_unc])

# Process noise uncertainty
Q = 1e-2 * Diagonal([1e-2, 1e-2, 1e-3, 1e-4, 1e-2, 1e-2, 1e-3, 1e-4])


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
Ts = 20
num_games = 1
num_particles = 50

p_transition = 0.98
p_init = 0.3


threshold = 0.1
max_iters = 50
step_size = 0.01

# Generate the ground truth.
costs = [QuadraticCostWithOffset(costs[1]), QuadraticCostWithOffset(costs[2])]

# leader_idx=2
gt_silq_num_runs=1

# config variables
gt_silq_threshold=0.001
gt_silq_max_iters=1000
gt_silq_step_size=1e-2
gt_silq_verbose=true


# Set initial controls so that we can solve a Stackelberg game with SILQGames.
us_init = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
# for ii in 1:num_players

# angular velocities
us_init[1][1,:] .= -.03
us_init[2][1,:] .= -.01

# accelerations
us_init[1][2,:] .= -.3
us_init[2][2,:] .= .3


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
