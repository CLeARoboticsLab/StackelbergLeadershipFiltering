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
