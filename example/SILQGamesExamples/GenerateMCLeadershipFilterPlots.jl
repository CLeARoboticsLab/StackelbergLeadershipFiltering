# A script that takes as input a JLD file containing S SILQGames trajectories and makes cumulative and individual plots.
using StackelbergControlHypothesesFiltering

using ProgressBars
using Dates
using Plots

using JLD

include("MCFileLocations.jl")
include("SILQGamesMCUtils.jl")

mc_silq_foldername = joinpath(mc_folder, data_folder)
mc_silq_jld_name = lf_data_file
input_data_path = joinpath(mc_silq_foldername, mc_silq_jld_name)

derivative_plots_folder = "plots_$(get_date_str())"
plots_path = joinpath(mc_silq_foldername, derivative_plots_folder)
isdir(plots_path) || mkdir(plots_path)

num_buckets = 100

data = load(input_data_path)["data"]

dyn = data["dyn"]
T = data["silq"]["T"]
times = data["times"]

num_sims = data["num_sims"]
all_xs = data["silq"]["xs"]
all_us = data["silq"]["us"]

gt_leader_idx = data["silq"]["gt_leader_idx"]
all_leader_idxs = data["all_particle_leader_idxs"] # second index is time
all_particle_xs = data["all_particle_xs"] # second index is time
all_particle_num_iterations = data["all_particle_num_iterations"]

mc_threshold = data["silq"]["threshold"]
mc_max_iters = data["silq"]["max_iters"]

num_particles = data["num_particles"]

all_probs = data["leadership_probs"]
all_x̂s = data["xests"]
all_P̂s = data["Pests"]
all_zs = data["measurements"]


# LQ
times_of_note = [2, 22, 122]

# Non-LQ
# times_of_note = [2, 22, 122]


for iter in ProgressBar(1:num_sims)
    folder_name = joinpath(plots_path, string(iter))
    isdir(folder_name) || mkdir(folder_name)

    true_xs = all_xs[iter, :, :]
    true_us = [@view all_us[jj][iter, :, :] for jj in 1:num_agents(dyn)]

    # Only needs to be generated once.
    p1a_i = plot_leadership_filter_positions(dyn, all_xs[iter, :, 1:T], all_x̂s[iter, :, 1:T])
    p1a_ii = plot_leadership_filter_measurements(dyn, all_xs[iter, :, 1:T], all_zs[iter, :, 1:T]; show_meas_annotation='d')

    pos_main_filepath = joinpath(folder_name, "lf_positions_main_L$(gt_leader_idx).pdf")
    pos_meas_filepath = joinpath(folder_name, "lf_positions_measurements_L$(gt_leader_idx).pdf")
    savefig(p1a_i, pos_main_filepath)
    savefig(p1a_ii, pos_meas_filepath)

    prob_plot = make_probability_plots(times[1:T], all_probs[iter, 1:T]; t_idx=times_of_note, include_gt=gt_leader_idx, player_to_plot=nothing)
    plot!(prob_plot, title="")
    prob_filepath = joinpath(folder_name, "lf_probs_L$(gt_leader_idx).pdf")
    savefig(prob_plot, prob_filepath)

    snapshot_freq = Int((T - 1)/20)
    jj = 0
    letter = 'e'
    for t in 2:snapshot_freq:T
        jj += 1

        p1b = plot_leadership_filter_measurement_details(dyn, all_leader_idxs[iter, t, :], num_particles, all_particle_num_iterations[iter, t, :], all_particle_xs[iter, t, :, :, :], true_xs, all_x̂s[iter, :, :]; t=t, letter=letter)
        if t in times_of_note
            letter = letter + 1
        end

        pos2_filepath = joinpath(folder_name, "0$(jj)_lf_t$(t)_positions_detail_L$(gt_leader_idx).pdf")
        savefig(p1b, pos2_filepath)
    end
end


# ############################
# #### Make the MC Probability plots. ####
# ############################
mean_probs = mean(all_probs, dims=[1])[1, :]
stddev_probs = (size(all_probs, 1) > 1) ? std(all_probs, dims=[1])[1, :] : zeros(T)

# Make the stddev bounds.
lower_p1 = min.(mean_probs .- 0, stddev_probs)
upper_p1 = min.(1 .- mean_probs, stddev_probs)

plot_unc = make_probability_plots(times[1:T], mean_probs[1:T]; include_gt=gt_leader_idx, stddevs=(lower_p1, upper_p1), t_idx=nothing)
plot!(plot_unc, title="")

filename = joinpath(plots_path, string("lf_mc$(num_sims)_L$(gt_leader_idx)_probs.pdf"))
savefig(plot_unc, filename)



