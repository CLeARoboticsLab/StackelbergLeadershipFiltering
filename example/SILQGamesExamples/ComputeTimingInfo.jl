using Dates
using LinearAlgebra
using JLD
using Statistics

include("MCFileLocations.jl")


lqp1_data_folder, s, lqp1_lf_path = get_final_lq_paths_p1()
# lqp2_data_folder, _, lqp2_lf_path = get_final_lq_paths_p2()
# uqp1_data_folder, _, uqp1_lf_path = get_final_uq_paths_p1()
nonlqp2_data_folder, nonlqp2_silq_path, nonlqp2_lf_path = get_final_nonlq_paths_p2()

function compute_silqgames_timing_info(data_folder, silq_filename)
    mc_foldername = joinpath(mc_folder, data_folder)
    input_data_path = joinpath(mc_foldername, silq_filename)
    silq_data = load(input_data_path)["data"]

    # Overall - includes compile times
    silq_mean = mean(silq_data["elapsed_times"])
    silq_std = std(silq_data["elapsed_times"])

    # Per-iteration times - isolate all the iterations and compute
    num_iterations = silq_data["num_iterations"]
    @assert !any(iszero.(num_iterations))

    mean_num_iters = mean(num_iterations .- 1)
    std_num_iters = std(num_iterations .- 1)

    iteration_times = silq_data["elapsed_iteration_times"]
    iters_of_interest = iteration_times[iteration_times[:] .!= 0]
    @assert length(iters_of_interest)-silq_data["num_sims"] == sum(num_iterations.-1) "length 1: $(length(iters_of_interest)-silq_data["num_sims"]), length 2: $(sum(num_iterations.-1))"

    mean_iter_time = mean(iters_of_interest)
    std_iter_time = std(iters_of_interest)

    return mean_num_iters, std_num_iters, mean_iter_time, std_iter_time
end

function compute_leadership_filter_timing_info(data_folder, lf_filename)
    mc_foldername = joinpath(mc_folder, data_folder)
    input_data_path = joinpath(mc_foldername, lf_filename)
    lf_data = load(input_data_path)["data"]

    # Overall - includes compile times
    lf_mean = mean(lf_data["elapsed_times"])
    lf_std = std(lf_data["elapsed_times"])

    # All particles per time step.
    timings = lf_data["all_lf_iter_timings"]
    @assert lf_data["silq"]["T"] == size(timings, 2) "length 1: $(lf_data["silq_data"]["T"]), length 2: $(size(timings, 2))"
    @assert !any(iszero.(timings))
    println(size(timings))

    mean_times_per_step = mean(timings; dims=[1])
    std_times_per_step = std(timings; dims=[1])

    return mean_times_per_step, std_times_per_step, mean(timings), std(timings)
end

function compute_overall_timing_info(data_folder, lf_filename)
    mc_foldername = joinpath(mc_folder, data_folder)
    input_data_path = joinpath(mc_foldername, lf_filename)
    lf_data = load(input_data_path)["data"]
    silq_data = lf_data["silq"]

    # LF times
    lf_mean = mean(lf_data["elapsed_times"])
    lf_std = std(lf_data["elapsed_times"])
    silq_mean = mean(silq_data["elapsed_times"])
    silq_std = std(silq_data["elapsed_times"])

    return lf_mean, lf_std, silq_mean, silq_std
end

# LQ P1
lq1_lf_m, lq1_lf_s, lq1_silq_m, lq1_silq_s = compute_overall_timing_info(lqp1_data_folder, lqp1_lf_path)
println("LQ SILQGames (P1), 201 timesteps @ 0.05s: $(lq1_silq_m) ± $(lq1_silq_s)")
println("LQ LF (P1), 201 timesteps @ 0.05s, Ts=30, Ns=50: $(lq1_lf_m) ± $(lq1_lf_s)")

# # LQ P1
# nonlq2_lf_m, nonlq2_lf_s, nonlq2_silq_m, nonlq2_silq_s = compute_overall_timing_info(nonlqp2_data_folder, nonlqp2_lf_path)
# println("NonLQ SILQGames (P2), 251 timesteps @ 0.02s: $(nonlq2_silq_m) ± $(nonlq2_silq_s)")
# println("NonLQ LF (P2), 251 timesteps @ 0.02s, Ts=30, Ns=100: $(nonlq2_lf_m) ± $(nonlq2_lf_s)")


lq_iters_mean, lq_iters_std, lq_iter_time_mean, lq_iter_time_std = compute_silqgames_timing_info(lqp1_data_folder, lqp1_silq_path)
lq_lf_iters_means, lq_lf_iters_stds, lq_lf_total_mean, lq_lf_total_std = compute_leadership_filter_timing_info(lqp1_data_folder, lqp1_lf_path)

nonlq_iters_mean, nonlq_iters_std, nonlq_iter_time_mean, nonlq_iter_time_std = compute_silqgames_timing_info(nonlqp2_data_folder, nonlqp2_silq_path)
