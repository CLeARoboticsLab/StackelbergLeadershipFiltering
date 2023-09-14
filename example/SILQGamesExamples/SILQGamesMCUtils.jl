using Base.Threads
using BenchmarkTools
using Dates
using LaTeXStrings
using LinearAlgebra
using Plots
using Statistics
using StatsBase

#### INITIALIZATION ####
function get_initial_conditions_at_idx(dyn::LinearDynamics, iter, num_sims, p1_angle, p1_magnitude, init_x₁; angle_range=(0, 2*pi))
    angle_diff = (angle_range[2] - angle_range[1])*((iter-1)//num_sims)
    us₁ = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
    xi₁ = deepcopy(init_x₁)
    new_angle = wrap_angle(angle_range[1] + angle_diff)
    # println("ANGLE LinDyn: ", new_angle, ", range 1: ", angle_range[1], ", diff: ", angle_diff)
    # new_angle = wrap_angle(p1_angle + angle_diff)
    xi₁[[xidx(dyn, 2), yidx(dyn, 2)]] = p1_magnitude * [cos(new_angle); sin(new_angle)]
    println("$iter - new IC: $xi₁")
    return xi₁, us₁
end

function get_initial_conditions_at_idx(dyn::UnicycleDynamics, iter, num_sims, p1_angle, p1_magnitude, init_x₁; angle_range=(0, 2*pi))
    angle_diff = (angle_range[2] - angle_range[1])*((iter-1)//num_sims)
    us₁ = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
    # us₁[1][2, 1:31] .= 1//2
    # us₁[2][2, 1:31] .= 1//2
    # us₁[1][2, 61:91] .= -1//2
    # us₁[2][2, 61:91] .= -1//2
    xi₁ = deepcopy(init_x₁)
    # new_angle = wrap_angle(p1_angle + angle_diff)
    new_angle = wrap_angle(angle_range[1] + angle_diff)
    # println("ANGLE UniDyn: ", new_angle, ", range 1: ", angle_range[1], ", diff: ", angle_diff)
    xi₁[[xidx(dyn, 2), yidx(dyn, 2)]] = p1_magnitude * [cos(new_angle); sin(new_angle)]

    # Set headings to be pointed towards the middle.
    xi₁[3] = wrap_angle(p1_angle - pi)
    xi₁[7] = wrap_angle(new_angle - pi)
    println("$iter - new IC: $xi₁")
    return xi₁, us₁
end



#### RUN SIMS + GENERATE DATA ####
function simulate_silqgames(num_sims, leader_idx, sg_obj, times, x₁; angle_range=(0, 2*pi))
    elapsed_times = zeros(num_sims)

    # Nominal trajectory is always zero-controls. x₁ is drawn as follows: P1 starts at (2, 1) unmoving and P2 rotates in a circle about the origin at the same radius.
    x1 = x₁[xidx(dyn, 1)]
    y1 = x₁[yidx(dyn, 1)]

    p1_angle = atan(y1, x1)
    p1_magnitude = norm([x1, y1])

    # Run MC sims.
    sim_iters = ProgressBar(1:num_sims)
    x1s = zeros(num_sims, xdim(dyn))
    u1s = [zeros(num_sims, udim(dyn, ii), T) for ii in 1:num_players]
    # for iter in sim_iters
    Threads.@threads for iter in sim_iters
        new_x₁, new_us_1 = get_initial_conditions_at_idx(dyn, iter, num_sims, p1_angle, p1_magnitude, x₁; angle_range=angle_range)
        elapsed_times[iter] = @elapsed begin
            xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times, new_x₁, new_us_1; manual_idx=iter)
        end

        x1s[iter, :] = new_x₁
        u1s[1][iter, :, :] = new_us_1[1]
        u1s[2][iter, :, :] = new_us_1[2]
    end
    # The @sync block ensures that all threads finish their iterations before continuing
    Threads.@sync begin
        println("\nCompleted SILQGames Simulation.")
        # Further code you want to execute after all threads finish
    end
    return sg_obj, x1s, u1s, elapsed_times
end


function generate_silq_jld_data(sg, leader_idx, times, dt, horizon, x1s, u1s, elapsed_times)
    timestamp = Dates.now()
    silq_data = Dict("timestamp" => timestamp,
                "times" => times,
                "T" => horizon,
                "dt" => dt,
                "t0" => times[1],
                # "silq_obj" => sg, # Stackelberg ILQGames object contains 
                "x1s" => x1s,
                "u1s" => u1s,
                "xs" => sg.xks,
                "us" => sg.uks,
                "gt_leader_idx" => leader_idx,
                "num_iterations" => sg.num_iterations,

                "num_sims" => sg.num_runs,
                "dyn" => sg.dyn,
                # "costs" => costs,
                "threshold" => sg.threshold,
                "max_iters" => sg.max_iters,

                "step_size" => sg.step_size,      # initial step size α₀
                "ss_reduce" => sg.ss_reduce,          # reduction factor τ
                "α_min" => sg.α_min,              # min step size
                "max_linesearch_iters" => sg.max_linesearch_iters,
                
                "convergence_metrics" => sg.convergence_metrics,
                "evaluated_costs" => sg.evaluated_costs,
                "elapsed_times" => elapsed_times,
                "elapsed_iteration_times" => sg.timings
                )
    return silq_data
end

function simulate_lf_with_silq_results(num_sims, leader_idx, dyn, prob_transition, T,
                                       times, all_true_xs, all_true_us, init_cov_P, prob_init,
                                       num_particles, Ts, num_games, R_meas, Q_proc,
                                       lf_threshold, lf_max_iters, lf_step_size,
                                       rng, silq_dict)
    elapsed_times = zeros(num_sims)

    discrete_state_transition, state_trans_P = generate_discrete_state_transition(prob_transition, prob_transition)
    s_init_distrib = Bernoulli(prob_init)
    process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q_proc)

    all_probs = -1 * ones(num_sims, T)
    all_x̂s = zeros(num_sims, xdim(dyn), T)
    all_P̂s = zeros(num_sims, xdim(dyn), xdim(dyn), T)
    all_zs = zeros(num_sims, xdim(dyn), T)

    # TODO(Debug what effect the extra 2 iterations from LF is having on results)
    # TODO(Why the extra Ts dimension?)
    all_particle_xs = zeros(num_sims, T, num_particles+2, xdim(dyn), Ts+1)
    all_particle_leader_idxs = zeros(num_sims, T, num_particles+2)
    all_particle_num_iterations = zeros(num_sims, T, num_particles+2)

    all_lf_iter_timings = zeros(num_sims, T)

    is_completed = [false for _ in 1:num_sims]

    sim_iters = ProgressBar(1:num_sims)
    # for ss in sim_iters
    Threads.@threads for ss in 1:num_sims
        # folder_name = joinpath(topfolder_name, string(ss))
        # isdir(folder_name) || mkdir(folder_name)

        # Extract states and controls from simulation.
        true_xs = all_true_xs[ss, :, :]
        true_us = [all_true_us[ii][ss, :, :] for ii in 1:num_players]

        # Augment the remaining states so we have T+Ts-1 of them.
        xs = hcat(true_xs, zeros(xdim(dyn), Ts-1))
        us = [hcat(true_us[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

        zs = zeros(xdim(dyn), T)

        # Fill in z as noisy state measurements.
        for tt in 1:T
            zs[:, tt] = rand(rng, MvNormal(true_xs[:, tt], R_meas))
        end

        all_zs[ss, :, :] = zs

        elapsed_times[ss] = @elapsed begin
            try 
                all_x̂s[ss, :, :], all_P̂s[ss, :, :, :], all_probs[ss, :], pf, sgs, all_lf_iter_timings[ss, :] = leadership_filter(dyn, costs, t0, lf_times,
                                   T,         # simulation horizon
                                   Ts,        # horizon over which the stackelberg game should be played,
                                   num_games, # number of stackelberg games played for measurement
                                   xs[:, 1],  # initial state at the beginning of simulation
                                   init_cov_P,        # initial covariance at the beginning of simulation
                                   us,        # the control inputs that the actor takes
                                   zs,        # the measurements
                                   R_meas,
                                   process_noise_distribution,
                                   s_init_distrib,
                                   discrete_state_transition;
                                   threshold=lf_threshold,
                                   rng,
                                   max_iters=lf_max_iters,
                                   step_size=lf_step_size,
                                   Ns=num_particles,
                                   verbose=false)
                is_completed[ss] = true
                for tt in 1:T
                    all_particle_leader_idxs[ss, tt, :] = sgs[tt].leader_idxs
                    all_particle_num_iterations[ss, tt, :] = sgs[tt].num_iterations
                    all_particle_xs[ss, tt, :, :, :] = sgs[tt].xks[:, :, :]
                end
            catch e
                println("\n\nLF simulation $ss from start position $(xs[:, 1]) failed.")
                println("$e\n\n")
                for tt in 1:T
                    all_particle_leader_idxs[ss, tt, :] .= 0
                    all_particle_num_iterations[ss, tt, :] .= -1
                    # all_particle_xs[ss, tt, :, :, :] .=  xs[:, 1] # stays 0
                end
            end
        end
    end

    # The @sync block ensures that all threads finish their iterations before continuing
    Threads.@sync begin
        println("\nCompleted leadership filter simulations $(sum(is_completed))/$(num_sims).")
        # Further code you want to execute after all threads finish
    end

    lf_data = Dict("timestamp" => silq_dict["timestamp"],
                "silq" => silq_dict,
                "silq_path" => mc_silq_filepath,
                "is_completed" => is_completed,

                "times" => lf_times,
                "dt" => dt,
                "t0" => times[1],

                "leadership_probs" => all_probs,
                "xests" => all_x̂s,
                "Pests" => all_P̂s,
                "measurements" => all_zs,

                "num_sims" => silq_dict["num_sims"],
                "dyn" => dyn,
            # "costs" => costs,

                "lf_threshold" => sg.threshold,
                "lf_max_iters" => sg.max_iters,

                "lf_step_size" => sg.step_size,      # initial step size α₀
                "lf_ss_reduce" => sg.ss_reduce,      # reduction factor τ
                "lf_α_min" => sg.α_min,              # min step size
                "lf_max_linesearch_iters" => sg.max_linesearch_iters,

                "cov_init" => init_cov_P,

            # Process noise uncertainty
                "process_noise" => Q_proc,

            # CONFIG:

                "meas_unc" => R_meas,
                "Ts" => Ts,
                "num_games" => num_games,
                "num_particles" => num_particles,
                "all_particle_leader_idxs" => all_particle_leader_idxs,
                "all_particle_xs" => all_particle_xs,
                "all_particle_num_iterations" => all_particle_num_iterations,

                "p_transition" => prob_transition,
                "p_init" => prob_init,

                "lf_threshold" => lf_threshold,
                "lf_max_iters" => lf_max_iters,
                "lf_step_size" => lf_step_size,

                "elapsed_times" => elapsed_times,
                "all_lf_iter_timings" => all_lf_iter_timings
                )

    # return all_probs, all_x̂s, all_P̂s, all_zs, all_particle_leader_idxs, all_particle_num_iterations, all_particle_xs
    return lf_data
end

#### METRICS + PLOTTING ####
function get_avg_convergence_w_uncertainty(all_conv_metrics, num_iterations, max_iters)
    curr_iters=1
    should_continue = true

    mean_metrics = zeros(max_iters)
    std_metrics = zeros(max_iters)

    while true
        idx_list = num_iterations .≥ curr_iters
        should_continue = curr_iters ≤ max_iters && any(idx_list)
        if !should_continue
            break
        end

        conv_metrics = all_conv_metrics[idx_list, 1, curr_iters]

        mean_metrics[curr_iters] = mean(conv_metrics)
        std_metrics[curr_iters] = (sum(idx_list) > 1) ? std(conv_metrics) : 0.

        curr_iters = curr_iters + 1
    end

    return mean_metrics[1:curr_iters-1], std_metrics[1:curr_iters-1], curr_iters-1
end

function plot_convergence(conv_metrics, num_iterations, max_iters, threshold; lower_bound=0.0, upper_bound=Inf)
    convergence_plot = get_standard_plot()
    plot!(yaxis=:log, xlabel="# Iterations", ylabel="Max Abs. State Difference")
    means, stddevs, final_idx = get_avg_convergence_w_uncertainty(conv_metrics, num_iterations, max_iters)
    conv_x = cumsum(ones(final_idx)) .- 1

    # conv_sum = conv_metrics[1, 1:num_iters] #+ conv_metrics[2, 1:num_iters]
    # for ii in 1:num_sims
    #     plot!(convergence_plot, conv_x, sg_obj.convergence_metrics[ii, 1, 1:num_iters_all+1], label="", color=:green)
    # end

    fill_α = 0.3
    lower = min.(means .- lower_bound, stddevs)
    upper = min.(upper_bound .- means, stddevs)
    # println(means, lower, upper)

    if final_idx > 2
        plot!(convergence_plot, conv_x, means, label=L"Mean $\ell_{\infty}$ Merit", color=:green, ribbon=(lower, upper), fillalpha=0.3, linewidth=3)
    else
        println("Lower: $(lower_bound), Upper: $(upper_bound)")
        lower_scatter = max.(lower_bound, means .- stddevs)
        upper_scatter = min.(upper_bound, means .+ stddevs)
        println("Lower Scatter: $(lower_scatter), Upper Scatter: $(upper_scatter)")
        println("Mean: $means")
        scatter!(convergence_plot, conv_x, means, yerr=(lower_scatter, upper_scatter), label=L"Mean $\ell_{\infty}$ Merit", color=:green, elinewidth=3, xticks=[0, 1])
    end
    plot!(convergence_plot, [0, final_idx-1], [threshold, threshold], label="Threshold", color=:purple, linestyle=:dot, linewidth=3)

    return convergence_plot
end

function plot_convergence_histogram(num_iterations, max_iters; num_bins=:auto)
    num_sims = length(num_iterations)
    if all(num_iterations .== 2)
        return histogram(num_iterations .- 1, bins=range(0.5, 1.5, step=1), xticks=[1], legend=false, ylabel="Frequency", xlabel="Iterations to Convergence")
    end
    hist = histogram(num_iterations .- 1, nbins=num_sims, legend=false, yticks=range(0, num_sims, step=1), ylabel="Frequency", xlabel="Iterations to Convergence")
    vline!(hist, [max_iters], label="Max Iterations", color=:black, linewidth=3)
    return hist
end

function plot_distance_to_origin(dyn, times, all_xs; lower_bound=0., upper_bound=Inf)
    num_players = num_agents(dyn)
    num_sims = size(all_xs, 1)
    T = size(all_xs, 3)

    all_dists_to_origin = zeros(num_sims, num_players, T)
    for ss in 1:num_sims
        for ii in 1:num_players
            traj = @view all_xs[ss, [xidx(dyn, ii), yidx(dyn, ii)], :]
            all_dists_to_origin[ss, ii, :] = [norm(pos_vec) for pos_vec in eachcol(traj)]
        end
    end
    mean_dists_to_origin = mean(all_dists_to_origin, dims=[1])[1, :, :]
    stddev_dists_to_origin = std(all_dists_to_origin, dims=[1])[1, :, :]

    lower1 = min.(mean_dists_to_origin[1, :] .- lower_bound, stddev_dists_to_origin[1, :])
    upper1 = min.(upper_bound .- mean_dists_to_origin[1, :], stddev_dists_to_origin[1, :])
    lower2 = min.(mean_dists_to_origin[2, :] .- lower_bound, stddev_dists_to_origin[2, :])
    upper2 = min.(upper_bound .- mean_dists_to_origin[2, :], stddev_dists_to_origin[2, :])

    d1 = get_standard_plot()
    plot!(xlabel="Time (s)", ylabel="Distance to Origin (m)")
    plot!(d1, times, mean_dists_to_origin[1, :], ribbon=(lower1, upper1), fillalpha=0.3, color=:red, label=L"\mathcal{A}_1", linewidth=3)
    plot!(d1, times, mean_dists_to_origin[2, :], ribbon=(lower2, upper2), fillalpha=0.3, color=:blue, label=L"\mathcal{A}_2", linewidth=3)

    return d1
end

function plot_distance_to_agents(dyn, times, all_xs; lower_bound=0., upper_bound=Inf)
    num_sims = size(all_xs, 1)
    T = size(all_xs, 3)

    all_dists_to_agent = zeros(num_sims, T)
    for ss in 1:num_sims
        traj = @view all_xs[ss, [xidx(dyn, 1), yidx(dyn, 1), xidx(dyn, 2), yidx(dyn, 2)], :]
        dist_traj = traj[1:2, :] - traj[3:4, :]
        all_dists_to_agent[ss, :] = [norm(dist_vec) for dist_vec in eachcol(dist_traj)]
    end
    mean_dists_to_agent = mean(all_dists_to_agent, dims=[1])[1, :]
    stddev_dists_to_agent = std(all_dists_to_agent, dims=[1])[1, :]

    lower = min.(mean_dists_to_agent .- lower_bound, stddev_dists_to_agent)
    upper = min.(upper_bound .- mean_dists_to_agent, stddev_dists_to_agent)

    d2 = get_standard_plot()
    plot!(xlabel="Time (s)", ylabel="Distance Between Agents (m)")
    plot!(d2, times, mean_dists_to_agent, ribbon=(lower, upper), fillalpha=0.3, color=:purple, label="", linewidth=3)
    return d2
end
