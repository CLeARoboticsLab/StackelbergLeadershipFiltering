using Distributions
using LinearAlgebra

# Set the random seed.
seed = 1

# Generative configuration - the data starts at an initial state (t0, v0, x0) and proceeds with 1D kinematic dynamics.
#                            At each time step, the generated data includes the state which dictates the direction of
#                            acceleration.

# Set parameters for data generation.
dt = 0.01
num_data = 501
num_cols = 5

only_pos_measurements = false


# The true generative probability of being in state 1 - used to generate the truth data.
true_prob_state1 = 0.7

# Acceleration magnitude for both state 1 and 2 (which are in opposite directions). Used to generate true data and dynamics.
accel_magnitude = 3.

# Set the initial state manually.
t0 = 0.0
init_direction = 1.0
v_mag_0 = -0.0
v0 = init_direction * v_mag_0 # true initial velocity point away from acceleration
x0 = 0.0                  # true initial position

# Define the measurement distribution uncertainties for true and measured data.
true_vel_meas_stdev = 0.1
true_pos_meas_stdev = 0.2

meas_vel_meas_stdev = true_vel_meas_stdev
meas_pos_meas_stdev = true_pos_meas_stdev

# Markov chain state transition probabilities - used to define the discrete state transition.
p₁₁ = 0.9
p₂₂ = 0.1


## RUNNING THE MODEL

# Define prior state gaussian parameters
x̄_prior = [v0, x0]                  # mean of the prior
P_prior = Diagonal([0.1^2, 0.2^2])  # covariance of the prior

# Specify the initial discrete state prior distribution.
p_init = 0.5
s_prior_distrib = Bernoulli(p_init)

# Specify the process noise.
v_bar = zeros(2)
Q = Diagonal([0.1^2, 0.2^2])
process_noise_distribution = MvNormal(v_bar, Q)
