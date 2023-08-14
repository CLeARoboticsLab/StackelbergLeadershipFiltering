include("MCFileLocations.jl")


lqp1_data_folder, _, lqp1_lf_path = get_final_lq_paths_p1()
lqp2_data_folder, _, lqp2_lf_path = get_final_lq_paths_p2()
uqp1_data_folder, _, uqp1_lf_path = get_final_uq_paths_p1()
nonlqp2_data_folder, _, nonlqp2_lf_path = get_final_nonlq_paths_p2()

function compute_timing_info(data_folder, lf_filename)
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
lq1_lf_m, lq1_lf_s, lq1_silq_m, lq1_silq_s = compute_timing_info(lqp1_data_folder, lqp1_lf_path)
println("LQ SILQGames (P1), 201 timesteps @ 0.05s: $(lq1_silq_m) ± $(lq1_silq_s)")
println("LQ LF (P1), 201 timesteps @ 0.05s, Ts=30, Ns=50: $(lq1_lf_m) ± $(lq1_lf_s)")

# LQ P1
uq1_lf_m, uq1_lf_s, uq1_silq_m, uq1_silq_s = compute_timing_info(uqp1_data_folder, uqp1_lf_path)
println("UQ SILQGames (P1), 201 timesteps @ 0.05s: $(uq1_silq_m) ± $(uq1_silq_s)")
println("UQ LF (P1), 201 timesteps @ 0.05s, Ts=30, Ns=50: $(uq1_lf_m) ± $(uq1_lf_s)")

# LQ P1
nonlq2_lf_m, nonlq2_lf_s, nonlq2_silq_m, nonlq2_silq_s = compute_timing_info(nonlqp2_data_folder, nonlqp2_lf_path)
println("NonLQ SILQGames (P2), 251 timesteps @ 0.02s: $(nonlq2_silq_m) ± $(nonlq2_silq_s)")
println("NonLQ LF (P2), 251 timesteps @ 0.02s, Ts=30, Ns=100: $(nonlq2_lf_m) ± $(nonlq2_lf_s)")
