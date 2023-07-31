using Plots
using Statistics
using StatsBase

#### INITIALIZATION ####
function get_initial_conditions_at_idx(dyn::LinearDynamics, iter, num_sims, p1_angle, p1_magnitude, init_x₁)
    angle_diff = 2*pi*((iter-1)//num_sims)
    us₁ = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
    xi₁ = deepcopy(init_x₁)
    new_angle = wrap_angle(p1_angle + angle_diff)
    xi₁[[xidx(dyn, 2), yidx(dyn, 2)]] = p1_magnitude * [cos(new_angle); sin(new_angle)]
    # println("$iter - new IC: $xi₁")
    return xi₁, us₁
end

function get_initial_conditions_at_idx(dyn::UnicycleDynamics, iter, num_sims, p1_angle, p1_magnitude, init_x₁)
    angle_diff = 2*pi*((iter-1)//num_sims)
    us₁ = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
    xi₁ = deepcopy(init_x₁)
    new_angle = wrap_angle(p1_angle + angle_diff)
    xi₁[[xidx(dyn, 2), yidx(dyn, 2)]] = p1_magnitude * [cos(new_angle); sin(new_angle)]

    # Set headings to be pointed towards the middle.
    xi₁[3] = wrap_angle(p1_angle - pi)
    xi₁[7] = wrap_angle(new_angle - pi)

    println("$iter - new IC: $xi₁")
    return xi₁, us₁
end


#### METRICS + PLOTTING ####
function get_avg_convergence_w_uncertainty(sg)
    curr_iters=1
    should_continue = true

    mean_metrics = zeros(sg.max_iters)
    std_metrics = zeros(sg.max_iters)

    while true
        idx_list = sg.num_iterations .≥ curr_iters
        should_continue = curr_iters ≤ sg.max_iters && any(idx_list)
        if !should_continue
            break
        end

        conv_metrics = sg.convergence_metrics[idx_list, 1, curr_iters]

        mean_metrics[curr_iters] = mean(conv_metrics)
        std_metrics[curr_iters] = (sum(idx_list) > 1) ? std(conv_metrics) : 0.

        curr_iters = curr_iters + 1
    end

    return mean_metrics[1:curr_iters-1], std_metrics[1:curr_iters-1], curr_iters-1
end

function plot_convergence(sg; lower_bound=-Inf, upper_bound=Inf)
    convergence_plot = get_standard_plot()
    plot!(yaxis=:log, xlabel="# Iterations", ylabel="Max Absolute State Difference")
    means, stddevs, final_idx = get_avg_convergence_w_uncertainty(sg)
    conv_x = cumsum(ones(final_idx)) .- 1

    # conv_sum = conv_metrics[1, 1:num_iters] #+ conv_metrics[2, 1:num_iters]
    # for ii in 1:num_sims
    #     plot!(convergence_plot, conv_x, sg_obj.convergence_metrics[ii, 1, 1:num_iters_all+1], label="", color=:green)
    # end

    fill_α = 0.3
    lower = min.(means .- lower_bound, stddevs)
    upper = min.(upper_bound .- means, stddevs)
    # println(means, lower, upper)

    plot!(convergence_plot, conv_x, means, label="Mean Merit Fn", color=:green, ribbon=(lower, upper), fillalpha=0.3)
    plot!(convergence_plot, [0, final_idx-1], [sg.threshold, sg.threshold], label="Threshold", color=:purple, linestyle=:dot)

    return convergence_plot
end

function plot_convergence_histogram(sg, num_bins)
    println("num iterations to converge: $(sg.num_iterations)")
    return histogram(sg.num_iterations .- 1, nbins=num_bins, legend=false, ylabel="Frequency", xlabel="Iterations to Convergence")
end

function plot_distance_to_origin(dyn, sg; lower_bound=0., upper_bound=Inf)
    num_players = num_agents(dyn)
    num_sims = sg.num_runs

    all_dists_to_origin = zeros(num_sims, num_players, T)
    for ss in 1:num_sims
        for ii in 1:num_players
            traj = @view sg.xks[ss, [xidx(dyn, ii), yidx(dyn, ii)], :]
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
    plot!(xlabel="Time (s)", ylabel="Dist. to Origin (m)")
    plot!(d1, times, mean_dists_to_origin[1, :], ribbon=(lower1, upper1), fillalpha=0.3, color=:red, label="P1")
    plot!(d1, times, mean_dists_to_origin[2, :], ribbon=(lower2, upper2), fillalpha=0.3, color=:blue, label="P2")

    return d1
end

function plot_distance_to_agents(dyn, sg; lower_bound=0., upper_bound=Inf)
    num_sims = sg.num_runs

    all_dists_to_agent = zeros(num_sims, T)
    for ss in 1:num_sims
        traj = @view sg.xks[ss, [xidx(dyn, 1), yidx(dyn, 1), xidx(dyn, 2), yidx(dyn, 2)], :]
        dist_traj = traj[1:2, :] - traj[3:4, :]
        all_dists_to_agent[ss, :] = [norm(dist_vec) for dist_vec in eachcol(dist_traj)]
    end
    mean_dists_to_agent = mean(all_dists_to_agent, dims=[1])[1, :]
    stddev_dists_to_agent = std(all_dists_to_agent, dims=[1])[1, :]

    lower = min.(mean_dists_to_agent .- lower_bound, stddev_dists_to_agent)
    upper = min.(upper_bound .- mean_dists_to_agent, stddev_dists_to_agent)

    d2 = get_standard_plot()
    plot!(xlabel="Time (s)", ylabel="Dist. Between Agents (m)")
    plot!(d2, times, mean_dists_to_agent, ribbon=(lower, upper), fillalpha=0.3, color=:purple, label="")
    return d2
end
