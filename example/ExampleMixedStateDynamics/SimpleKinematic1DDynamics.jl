######## File: SimpleKinematics1DDynamics.jl ########
using Distributions
using Random

include("ConfigSimpleKinematic1DDynamics.jl")


## Define functions
function generate_transition_matrices(a)
    # State transition matrix
    Φ(tᵢ,t₀) = [ exp(tᵢ - t₀)                            0;
                (1/2) * (tᵢ - t₀)^2 * exp(tᵢ - t₀)  exp(tᵢ - t₀)]

    # Control transition matrix
    G(tᵢ,t₀) = a * [ exp(tᵢ - t₀) - (tᵢ - t₀) - 1;
                    (1/2) * (exp(tᵢ - t₀) * (((tᵢ - t₀) - 4) * (tᵢ - t₀) + 8) - (((tᵢ - t₀) + 4) * (tᵢ - t₀)) - 8)][:,:]

    # Process noise transition matrix

    # for only position
    # Γ(tᵢ,t₀) = [ 0; exp(tᵢ - t₀) - 1]

    # For velocity and position
    Γ(tᵢ,t₀) = [ exp(tᵢ - t₀) - 1  0 ; 0  exp(tᵢ - t₀) - 1]

    return Φ, G, Γ
end

function generate_measurement_models(only_pos_measurements::Bool)
    # measurement functions
    h(x) = (only_pos_measurements) ? x[2] * ones(1) : x
    H(x) = (only_pos_measurements) ? [0 1] : [1 0; 0 1]
    return h, H
end

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

function generate_dynamics(a)

    # TODO(hamzah) - this derivation may have a bug in it that causes divergence in example script
    #              - it is left here in case it is needed again
    # Φ, G, Γ = generate_transition_matrices(a)
    # function f(t_range, x, u, v)
    #     t0 = t_range[1]
    #     t = t_range[2]
    #     @assert t ≥ t0

    #     return Φ(t, t0) * x + G(t, t0) * u + Γ(t, t0) * v
    # end

    i = 1

    A = [0 0; 1. 0]
    B = [1.; 0][:,:] # The multiplier here is the difference between states.
    D = [1. 0; 0 1.]
    cont_dyn = ContinuousLinearDynamics(A, [B], D)
    function f_dynamics(t_range, x, u, v)
        t0 = t_range[1]
        t = t_range[2]
        @assert t ≥ t0

        # no change if the times are the same.
        if t == t0 return x end

        dt = t - t0
        dyn = discretize(cont_dyn, dt)

        return propagate_dynamics(dyn, t_range, x, [a * u], v[:])
    end
    return f_dynamics
end

# random number generator
rng = MersenneTwister(seed)

# Generate data going right.
data = zeros(num_data, num_cols)
data[1, 1] = t0
data[1, 2] = v0
data[1, 3] = x0

a₁ = accel_magnitude
a₂ = -accel_magnitude
is_1 = rand(rng, Bernoulli(true_prob_state1), num_data)
accel = zeros(num_data)
accel[is_1] .= a₁
accel[.!is_1] .= a₂

f = generate_dynamics(1) # in this case, the acceleration will be provided as input.

n = zeros(size(data[:, 2:3]))
n[1, :] = [v0;x0]
for i in 2:num_data
    time = (i-1) * dt
    data[i, 1] = t0 + time

    # compute the time stepped version too, just to compare error - should be pretty small
    time_range = (time - dt, time)
    n[i, :] = f(time_range, n[i-1, :], [accel[i]], zeros(2))

    data[i, 2] = accel[i] * dt + data[i-1, 2]
    data[i, 3] = (1/2) * accel[i] * dt^2 + data[i-1, 2] * dt + data[i-1, 3]
end
println("final true dynamics: ", n[num_data, :])
println("final true state: ", data[num_data, 2], " ", data[num_data, 3])

# Make measurement distribution and generate measurement data.
if only_pos_measurements
    true_R = Diagonal([true_pos_meas_stdev^2])
    eps_distrib = MvNormal(zeros(1), true_R)
    data[:, 5] = data[:, 3] + rand(rng, eps_distrib, num_data)'
else
    true_R = Diagonal([true_vel_meas_stdev^2, true_pos_meas_stdev^2])
    eps_distrib = MvNormal(zeros(2), true_R)
    data[:, 4:5] = data[:, 2:3] + rand(rng, eps_distrib, num_data)'
end


t = data[:,1]           # times
z = (only_pos_measurements) ? data[:, 5] : data[:, 4:5]
x_true = data[:,2:3]    # true values

ℓ = size(z)[1]          # number of measurements

t0 = t[1]               # time of first measurement

# measurement error covariance
if only_pos_measurements
    meas_R = Diagonal([meas_pos_meas_stdev^2])
    eps_distrib = MvNormal(zeros(1), meas_R)
else
    meas_R = Diagonal([meas_vel_meas_stdev^2, meas_pos_meas_stdev^2])
    eps_distrib = MvNormal(zeros(2), meas_R)
end


# Generate the discrete state transition
discrete_state_transition, disc_state_matrix = generate_discrete_state_transition(p₁₁, p₂₂)

# Generate dynamics and measurement model for state 1
f₁ = generate_dynamics(a₁)
h₁, H₁ = generate_measurement_models(only_pos_measurements)

# Generate dynamics and measurement model for state 2
f₂ = generate_dynamics(a₂)
h₂, H₂ = generate_measurement_models(only_pos_measurements)

# saturation causes Nans?
