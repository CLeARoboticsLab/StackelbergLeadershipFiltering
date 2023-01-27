using Distributions, POMDPs, ParticleFilters, StackelbergControlHypothesesFiltering

MixedStateT = Tuple{Bool, AbstractVector{Float64}, Float64}
StateT = MixedStateT
CtrlT = AbstractVector{<:AbstractVector{Float64}}
ObservationT = MixedStateT

struct MixedPOMDP <: POMDP{StateT, CtrlT, ObservationT}
    dyns::AbstractVector{<:Dynamics}
    dt::Float64
    disc_state_trans_distribution::Bernoulli
    state0::MixedStateT
end

# state_trans_prob is probability of state transition to the other state, i.e. evaluating the bernoulli distrib to 1).
MixedPOMDP(dyns, dt, state_trans_prob::Float64, state0) = MixedPOMDP(dyns, dt, Bernoulli(state_trans_prob), state0)

# Gets a distribution of possible initial states of the POMDP.
POMDPs.initialstate(m::MixedPOMDP) = Dirac(m.state0)

# Observations get the state exactly.
function get_discrete_observation_distribution(m::MixedPOMDP, disc_state::Bool)
    return Dirac(disc_state)
end

function get_continuous_observation_distribution_array(m::MixedPOMDP, state::MixedStateT)
    dyn = state[1] ? m.dyns[1] : m.dyns[2]
    num_cont_states = xdim(dyn)

    # TODO(hamzah): Fix incorrect independence assumption - by doing this, I assume independence where it does not
    #               necessarily exist. Also, I have recompose the vector once the sampling occurs.
    return [Dirac(state[2][ii]) for ii in 1:num_cont_states]
end

POMDPs.observation(m::MixedPOMDP, state::MixedStateT) = product_distribution(
    [
        get_discrete_observation_distribution(m, state[1]),
        get_continuous_observation_distribution_array(m, state)...,
        Dirac(state[3])
    ]
)

function sample_observation(m::MixedPOMDP, sp::StateT, rng)
    raw_obs = rand(rng, observation(m, sp))
    num_x = length(raw_obs) - 2

    obs = (
        !iszero(raw_obs[1]),
        [Float64(raw_obs[i+1]) for i in 1:num_x],
        Float64(raw_obs[num_x+2])
    )
    println(obs)
    return obs
end

function POMDPs.gen(m::MixedPOMDP, state, ctrls, rng)
    # For this POMDP, a realization of 1 means that we have a state transition away from the current state.
    new_discrete_state = iszero(rand(rng, m.disc_state_trans_distribution)) ? state[1] : !state[1]

    # Select the dynamics that is appropriate based on the state.
    dyn = state[1] ? m.dyns[1] : m.dyns[2]

    xm1 = state[2]
    tm1 = state[3]
    sp = (
        new_discrete_state,
        propagate_dynamics(dyn, tm1, xm1, ctrls, zeros(vdim(dyn)))[:],
        tm1 + m.dt
    )

    o = sample_observation(m, sp, rng)
    r = 0.0

    out = (;sp, o, r)
    return out
end



d0 = false
x0 = ones(1)
t0 = 0.0
s0 = (d0, x0, t0)

dt = 1.0
p_transition = 0.5

# Both dynamics are zero-input systems.
dyns = [LinearDynamics(ones(1)[:,:], [zeros(1)[:,:] for ii in 1:2]),
        LinearDynamics(-ones(1)[:,:], [zeros(1)[:,:] for ii in 1:2])]

pomdp = MixedPOMDP(dyns, dt, p_transition, s0)
n_particles = 100
T = 50
pf = BootstrapFilter(pomdp, n_particles)
b = initialize_belief(pf, initialstate(pomdp))

st = rand(initialstate(pomdp))
us = [zeros(1) for ii in 1:2]
b_hist = [b.particles]
s_hist = [st]

i = 1
# for i ∈ 1:T-1
    global b, o, r, st
    st, o, r = @gen(:sp,:o,:r)(pomdp, st, us)
    b = update(pf, b, us, o)
    # push!(s_hist, st)
    # push!(b_hist, b.particles)
# end

