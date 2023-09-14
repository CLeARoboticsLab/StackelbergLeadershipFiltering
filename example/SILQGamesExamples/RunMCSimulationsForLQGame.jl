# A script that runs S simulations each of which solved SILQGames for an LQ game from different initial points around a circle
# and saves the results for each game to a jld file.

using StackelbergControlHypothesesFiltering

# using Base.Threads
using LinearAlgebra
using ProgressBars
# using Statistics
# using StatsBase

using Dates
using Distributions
using LaTeXStrings
using Random
using Plots

using JLD

include("SILQGamesMCUtils.jl")
include("LQ_parameters.jl")

costs = [QuadraticCostWithOffset(ss_costs[1]), QuadraticCostWithOffset(ss_costs[2])]
# costs = ss_costs
@assert tr(ss_costs[1].Rs[1]) == 2

# For player cost
# costs = [pc_cost_1, pc_cost_2]

num_sims = 100

data_folder = "mc_data"

topfolder_name = joinpath(data_folder, "lq_mc$(num_sims)_L$(leader_idx)_$(get_date_str())")
isdir(topfolder_name) || mkdir(topfolder_name)

# config variables
mc_threshold=1.5e-2
mc_max_iters=50
mc_step_size=1e-2
mc_verbose=false

# 1. Initialize the SILQGames object.
sg_obj = initialize_silq_games_object(num_sims, T, dyn, costs;
                                      state_reg_param=0.0, control_reg_param=1e-32, ensure_pd=false,
                                      threshold=mc_threshold, max_iters=mc_max_iters, step_size=mc_step_size, verbose=mc_verbose)

# 2. Run the Monte Carlo SILQGames simulation.
y2idx = yidx(dyn, 2)
x2idx = xidx(dyn, 2)
angle = atan(x₁[y2idx], x₁[x2idx])
angle_diff = 0.2
angle_range_lq=(angle-angle_diff, angle+angle_diff)
sg, x1s, u1s, silq_elapsed = simulate_silqgames(num_sims, leader_idx, sg_obj, times, x₁; angle_range=angle_range_lq)

# 3. Generate the data and save to the specified file.
silq_data = generate_silq_jld_data(sg, leader_idx, times, dt, T, x1s, u1s, silq_elapsed)
mc_silq_filepath = joinpath(topfolder_name, "lq_silq_mc$(num_sims)_L$(leader_idx)_th$(mc_threshold)_ss$(mc_step_size)_M$(mc_max_iters).jld")
save("$(mc_silq_filepath)", "data", silq_data)
println("Saved $(num_sims) SILQ simulations to $(mc_silq_filepath).")


# Leadership filtering.
t0 = times[1]
lf_times = dt * (cumsum(ones(2*T)) .- 1)
pos_unc = 1e-3
θ_inc = 1e-3
vel_unc = 1e-4
P₁ = Diagonal([pos_unc, pos_unc, θ_inc, vel_unc, pos_unc, pos_unc, θ_inc, vel_unc])

# Process noise uncertainty
Q = 1e-2/5 * Diagonal([1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4, 1e-2, 1e-4])

# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = 0.005 * Matrix(I, xdim(dyn), xdim(dyn))
Ts = 30
num_games = 1
num_particles = 50

p_transition = 0.98
p_init = 0.5

lf_threshold = 1.5e-2
lf_max_iters = 50
lf_step_size = 1e-2



# 4. Run the leadership filter simulation.
# all_probs, all_x̂s, all_P̂s, all_zs, all_particle_leader_idxs, all_particle_num_iterations, all_particle_xs =
lf_data = simulate_lf_with_silq_results(num_sims, leader_idx, sg.dyn, p_transition, T,
                                        lf_times, sg.xks, sg.uks, P₁, p_init,
                                        num_particles, Ts, num_games, R, Q,
                                        lf_threshold, lf_max_iters, lf_step_size,
                                        rng, silq_data)


# 5. Save to the specified file.
mc_lf_filepath = joinpath(topfolder_name, "lq_lf_mc$(num_sims)_L$(leader_idx)_th$(lf_threshold)_ss$(lf_step_size)_M$(lf_max_iters).jld")
save("$(mc_lf_filepath)", "data", lf_data)
println("Saved $(num_sims) LF simulations to $(mc_lf_filepath)
         generated from $(mc_silq_filepath).")
