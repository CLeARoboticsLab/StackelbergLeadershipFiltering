# TODO(hamzah) Implement the estimator from the project or find a package that does that.

# TODO(hamzah) Tests?

using Distributions
using Random

function initialize_weights(Ns)
    return 1/Ns * ones(Ns)
end

function compute_measurement_lkhd(distrib, measurement)
    return pdf(distrib, measurement)
end

function compute_effective_num_particles(weights)
    return 1 / sum(weights.^2)
end

function resample_condition(Ns, Ns_hat)
    return Ns_hat < 0.5 * Ns
end
export resample_condition

# Resample Ns times with replacement from the provided particles and weights.
function resample_particles(rng, Ns, particles, weights)
    num_states = size(particles, 1)
    𝒳_new = zeros(num_states, Ns)
    cumulative_weights = cumsum(weights)
    for i in 1:Ns
        j = findfirst(cumulative_weights .> rand(rng, Uniform(0,1)))
        𝒳_new[:, i] = particles[:, j]
    end
    return 𝒳_new
end

# Resample Ns times with replacement from the provided particles and weights.
function resample_particles_w_discrete_state(rng, Ns, particles, disc_state_particles, weights)
    num_states = size(particles, 1)
    num_disc_states = size(disc_state_particles, 1)
    𝒳_new = zeros(num_states, Ns)
    s_new = zeros(num_disc_states, Ns)
    cumulative_weights = cumsum(weights)
    for i in 1:Ns
        j = findfirst(cumulative_weights .> rand(rng, Uniform(0,1)))
        𝒳_new[:, i] = particles[:, j]
        s_new[:, i] = disc_state_particles[:, j]
    end
    return 𝒳_new, s_new
end

function sample_process_noise(rng, process_noise_distribution, Ns)
    return rand(rng, process_noise_distribution, Ns)
end

function sample_discrete_state(rng, s_distribution, num_disc_states, Ns)
    new_s = zeros(MultiStateIntT, num_disc_states, Ns)
    raw_s = rand(rng, s_distribution, Ns)
    new_s[:, isone.(raw_s)] .= 1
    new_s[:, iszero.(raw_s)] .= 2
    return new_s
end

# 0. Add arguments that include
#    - [DONE] multiple dynamics
#    - [DONE] (for later) multiple measurement models
#    - [DONE, same as prior for now] initial probability of each discrete state

# Q: How to represent the discrete state in places where the continuous state is needed?
# Assumption: discrete state takes value 1 or 2.
MultiStateIntT = Int8
function two_state_PF(x̄_prior,
                      P_prior,
                      u_inputs,
                      s_init_distribution::Distribution{Univariate, Discrete},
                      times,
                      t0,
                      z,
                      R,
                      discrete_state_transition::Function,
                      f_dynamics::AbstractVector{<:Function},
                      h_measures::AbstractVector{<:Function};
                      seed=1,
                      Ns=1000)

    num_disc_states = 1
    num_disc_state_values = 2
    @assert length(h_measures) == num_disc_state_values == length(f_dynamics)

    rng = MersenneTwister(seed) # random number generator

    ℓ = size(z, 1)
    ℓ = size(times, 1)
    n = size(x̄_prior, 1)
    m = size(R, 1)

    𝒳 = zeros(n,Ns,ℓ)
    𝒵 = zeros(m,Ns,ℓ)
    w = zeros(Ns,ℓ)
    x̂ = zeros(n,ℓ)
    z̄ = zeros(m,ℓ)
    P = zeros(n,n,ℓ)
    P̄_zz = zeros(m,m,ℓ)
    ϵ_bar = zeros(ℓ,m)
    ϵ_hat = zeros(ℓ,m)
    N̂s = zeros(ℓ)

    # [DONE] 1. Initialize vectors to track the discrete state for each particle at each time.
    s = zeros(MultiStateIntT, num_disc_states, Ns, ℓ)
    ŝ_probs = zeros(num_disc_states,ℓ)

    for k in 1:ℓ

        if k == 1
            # [DONE] 2a. Initialize the discrete state for each particle at the initial time.
            s_prev = sample_discrete_state(rng, s_init_distribution, num_disc_states, Ns)
            t_prev = t0

            prior_state_distrib = MvNormal(x̄_prior, P_prior)
            𝒳_prev = rand(rng, prior_state_distrib, Ns)
            weights_prev = initialize_weights(Ns)
        else
            # [DONE] 2b. Extract the discrete state for each particle from the previous timestep.
            s_prev = s[:,:,k-1]
            t_prev = times[k-1]

            𝒳_prev = 𝒳[:,:,k-1]
            weights_prev = w[:,k-1]
        end

        # compute dynamics and measurement likelihoods
        p = zeros(Ns)

        time_range = (t_prev, times[k])

        for i in 1:Ns
            # 3a. Update the dynamics propagation to use the `propagate_dynamics` function.
            # [DONE] 3b. The dynamics should be selected using the discrete state appropriately, but that can be done before
            #            passing to `propagate_dynamics`.
            # [DONE] 3c. Dynamics propagation should update the discrete state. We can use a Bernoulli for simiplicity for now,
            #            but we may eventually want a more complicated Markov Chain.
            s_probs_in = vcat(ŝ_probs[:, k], [1]-ŝ_probs[:, k])[:]
            s[:,i,k] = discrete_state_transition(time_range, s_prev[:,i], s_probs_in, 𝒳_prev[:,i], u_inputs[:,k], rng)

            # Resample the state for each particle and extract an index to select the dynamics.
            s_idx = s_prev[:,i][1]

            𝒳[:,i,k] = f_dynamics[s_idx](time_range, 𝒳_prev[:,i], u_inputs[:,k], rng)
            𝒵[:,i,k] = h_measures[s_idx](𝒳[:,i,k])

            distrib = MvNormal(𝒵[:,i,k], R)
            p[i] = compute_measurement_lkhd(distrib, z[k, :])
        end
        c_inv = weights_prev' * p

        # calculate weights and weighted empirical means
        x̂[:,k] = zeros(n)
        for i in 1:Ns
            w[i,k] = 1/c_inv * p[i] * weights_prev[i]
            x̂[:,k] += w[i,k] * 𝒳[:,i,k]

            # use previous weights since this is used for the pred residuals
            z̄[:,k] += weights_prev[i] * 𝒵[:,i,k]
        end

        # calculate empirical weighted covariances
        P[:,:,k] = sum(w[i,k] * (𝒳[:,i,k] - x̂[:,k]) * (𝒳[:,i,k] - x̂[:,k])'
                       for i in 1:Ns)
        P̄_zz[:,:,k] = sum(weights_prev[i] * (𝒵[:,i,k] - z̄[:,k]) * (𝒵[:,i,k] - z̄[:,k])'
                          for i in 1:Ns)

        # 5. [DONE] Ensure that resampling includes the discrete state.
        # resample if needed
        N̂s[k] = compute_effective_num_particles(w[:, k])
        if resample_condition(Ns, N̂s[k])
            # Resample Ns new particles from the current particles and weights.
            𝒳[:, :, k], s[:, :, k] = resample_particles_w_discrete_state(rng, Ns, 𝒳[:, :, k], s[:, :, k], w[:, k])
            w[:,k] = initialize_weights(Ns)
        end

        # 4. [DONE] Generate an empirical weighted probability of being in state 1 and store that.
        # TODO(hamzah) Generalize this for more than one discrete state when needed.
        # BUG?: Should this be before or after resampling?
        in_state_1 = isone.(s[:, :, k])[:]
        ŝ_probs[:, k] .= sum(w[in_state_1, k])

        # calculate residuals
        ϵ_bar[k,:] = z[k,:] - z̄[:,k]

        # 6. [DONE] Ensure that the call to h_meas uses the appropriate measurement likelihoods and weights.
        # ẑ is computed using the expected value of the measurement likelihoods of each dynamics model.
        p = ŝ_probs[k]
        ẑ = p * h_measures[1](x̂[:,k]) + (1-p) * h_measures[2](x̂[:,k])
        ϵ_hat[k,:] = z[k,:] - ẑ
    end

    # 7. [DONE] Return new states and probabilities from function.
    particles = 𝒳
    return x̂, P, z̄, P̄_zz, ϵ_bar, ϵ_hat, N̂s, s, ŝ_probs, particles
end
export two_state_PF


# TODO(hamzah) - refactor to put particle filter in its own folder/file with a helpers file
function PF(x̄_prior,
            P_prior,
            u_inputs,
            times,
            t0,
            z,
            R,
            f_dynamics::Function,
            h_meas::Function,
            process_noise_distribution;
            seed=1,
            Ns=1000)

    rng = MersenneTwister(seed) # random number generator

    ℓ = size(z, 1)
    n = size(x̄_prior, 1)
    m = size(R, 1)

    𝒳 = zeros(n,Ns,ℓ)
    𝒵 = zeros(m,Ns,ℓ)
    w = zeros(Ns,ℓ)
    x̂ = zeros(n,ℓ)
    z̄ = zeros(m,ℓ)
    P = zeros(n,n,ℓ)
    P̄_zz = zeros(m,m,ℓ)
    ϵ_bar = zeros(ℓ,m)
    ϵ_hat = zeros(ℓ,m)
    N̂s = zeros(ℓ)
    for k in 1:ℓ

         if k == 1
            t_prev = t0

            prior_state_distrib = MvNormal(x̄_prior, P_prior)
            𝒳_prev = rand(rng, prior_state_distrib, Ns)
            weights_prev = initialize_weights(Ns)
        else
            # [DONE] 2b. Extract the discrete state for each particle from the previous timestep.
            t_prev = times[k-1]

            𝒳_prev = 𝒳[:,:,k-1]
            weights_prev = w[:,k-1]
        end

        # compute dynamics and measurement likelihoods
        p = zeros(Ns)
        c_inv = 0.0
        time_range = (t_prev, times[k])
        vs = sample_process_noise(rng, process_noise_distribution, Ns)
        for i in 1:Ns
            𝒳[:,i,k] = f_dynamics(time_range, 𝒳_prev[:,i], u_inputs[:,k], rng)
            𝒵[:,i,k] = h_meas(𝒳[:,i,k])

            distrib = MvNormal(𝒵[:,i,k], R)
            p[i] = compute_measurement_lkhd(distrib, z[k, :])
            c_inv += p[i] * weights_prev[i]
        end

        # calculate weights and weighted empirical means
        x̂[:,k] = zeros(n)
        for i in 1:Ns
            w[i,k] = 1/c_inv * p[i] * weights_prev[i]
            x̂[:,k] += w[i,k] * 𝒳[:,i,k]
            
            # use previous weights since this is used for the pred residuals
            z̄[:,k] += weights_prev[i] * 𝒵[:,i,k]
        end

        # calculate empirical weighted covariances
        P[:,:,k] = sum(w[i,k] * (𝒳[:,i,k] - x̂[:,k]) * (𝒳[:,i,k] - x̂[:,k])' 
                        for i in 1:Ns)
        P̄_zz[:,:,k] = sum(weights_prev[i] * (𝒵[:,i,k] - z̄[:,k]) * (𝒵[:,i,k] - z̄[:,k])' 
                        for i in 1:Ns)

        # resample if needed
        N̂s[k] = compute_effective_num_particles(w[:, k])
        if resample_condition(Ns, N̂s[k])
            # Resample Ns new particles from the current particles and weights.
            𝒳[:, :, k] = resample_particles(rng, Ns, 𝒳[:, :, k], w[:, k])
            w[:,k] = initialize_weights(Ns)
        end

        # calculate residuals
        ϵ_bar[k,:] = z[k,:] - z̄[:,k]
        ϵ_hat[k,:] = z[k,:] - h_meas(x̂[:,k])

    end

    particles = 𝒳
    return x̂, P, z̄, P̄_zz, ϵ_bar, ϵ_hat, N̂s, particles
end
export PF
