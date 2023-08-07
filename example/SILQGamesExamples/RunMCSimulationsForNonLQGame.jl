# NOTE: Remember to change the cost of the LQ shepherd and sheep game.

using StackelbergControlHypothesesFiltering

using LinearAlgebra
using ProgressBars
using Statistics
using StatsBase

# include("nonLQ_parameters.jl")
include("DELETE_leadfilt_non_LQ_parameters.jl")
include("SILQGamesMCUtils.jl")

num_sims = 20

# config variables
mc_threshold=1e-3
mc_max_iters=2500
mc_step_size=1e-2
mc_verbose=false

# 1. Initialize the SILQGames object.
sg_obj = initialize_silq_games_object(num_sims, T, dyn, costs;
                                      threshold=mc_threshold, max_iters=mc_max_iters, step_size=mc_step_size, verbose=mc_verbose)

# 2. Run the Monte Carlo SILQGames simulation.
sg, x1s, u1s, silq_elapsed = simulate_silqgames(num_sims, leader_idx, sg_obj, times, x₁)

# 3. Generate the data and save to the specified file.
silq_data = generate_silq_jld_data(sg, leader_idx, times, dt, T, x1s, u1s, silq_elapsed)
mc_silq_filepath = joinpath(topfolder_name, "nonlq_silq_mc$(num_sims)_L$(leader_idx)_th$(mc_threshold)_ss$(mc_step_size)_M$(mc_max_iters).jld")
save("$(mc_silq_filepath)", "data", silq_data)
println("Saved $(num_sims) SILQ simulations to $(mc_silq_filepath).")


# Leadership filtering.
t0 = times[1]
lf_times = dt * (cumsum(ones(2*T)) .- 1)
# pos_unc = 1e-3
# θ_inc = 1e-3
# vel_unc = 1e-4
# P₁ = Diagonal([pos_unc, pos_unc, θ_inc, vel_unc, pos_unc, pos_unc, θ_inc, vel_unc])

# # Process noise uncertainty
# Q = 1e-2 * Diagonal([1e-2, 1e-2, 1e-3, 1e-4, 1e-2, 1e-2, 1e-3, 1e-4])

# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

# R = 0.01 * Matrix(I, xdim(dyn), xdim(dyn))
# Ts = 30
# num_games = 1
# num_particles = 100

# p_transition = 0.98
# p_init = 0.5


# 4. Run the leadership filter simulation.
# all_probs, all_x̂s, all_P̂s, all_zs, all_particle_leader_idxs, all_particle_num_iterations, all_particle_xs =
lf_data = simulate_lf_with_silq_results(num_sims, leader_idx, sg.dyn, p_transition, T,
                                        lf_times, sg.xks, sg.uks, P₁, p_init,
                                        num_particles, Ts, num_games, R, Q,
                                        lf_threshold, lf_max_iters, lf_step_size,
                                        rng, silq_data)


# 5. Save to the specified file.
mc_lf_filepath = joinpath(topfolder_name, "nonlq_lf_mc$(num_sims)_L$(leader_idx)_th$(lf_threshold)_ss$(lf_step_size)_M$(lf_max_iters).jld")
save("$(mc_lf_filepath)", "data", lf_data)
println("Saved $(num_sims) LF simulations to $(mc_lf_filepath)
         generated from $(mc_silq_filepath).")

