using StackelbergControlHypothesesFiltering

dt = 0.05
T = 301
t0 = 0.0
horizon = T * dt
# TODO(hamzah) - We do double the times as needed so that there's extra for the Stackelberg history. Make this tight.
times = dt * (cumsum(ones(2*T)) .- 1)

cont_lin_dyn = ShepherdAndSheepDynamics()
dyn = discretize(cont_lin_dyn, dt)
costs = ShepherdAndSheepCosts()
num_players = num_agents(dyn)

leader_idx = 1
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 0.; 1.; 0.; -1.; 0; 2; 0]
pos_unc = 1e-3
vel_unc = 1e-4
P₁ = Diagonal([pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc, pos_unc, vel_unc])

# Process noise uncertainty
Q = 1e-2 * Diagonal([1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4])


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
