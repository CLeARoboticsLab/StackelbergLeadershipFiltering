using Plots
using ProgressBars

include("PassingScenarioConfig.jl")

function add_lane_lines!(plt, cfg::PassingScenarioConfig, limits)
    plot!(plt, limits, [get_center_line_x(cfg), get_center_line_x(cfg)], ls=:dash, color=:black, label="")
    plot!(plt, limits, [get_right_lane_boundary_x(cfg), get_right_lane_boundary_x(cfg)], color=:black, label="")
    plot!(plt, limits, [get_left_lane_boundary_x(cfg), get_left_lane_boundary_x(cfg)], color=:black, label="")
    return plt
end

# This generates the plots ready to be placed into the paper.
function make_passing_scenario_pdf_plots(folder_name, snapshot_freq, cfg, limits, dyn, horizon, times, true_xs, true_us, probs, x̂s, zs, num_particles)
    T = horizon
    limits_tuple = tuple(limits...)
    iter1 = ProgressBar(2:snapshot_freq:T)

    rotated_true_xs = rotate_state(dyn, true_xs)
    rotated_zs = rotate_state(dyn, zs)
    rotated_x̂s = rotate_state(dyn, x̂s)
    rotate_particle_state(xs) = rotate_state(dyn, xs)

    # Only needs to be generated once.
    p1a = plot_leadership_filter_positions(sg_objs[1].dyn, rotated_true_xs[:, 1:T], rotated_x̂s[:, 1:T], rotated_zs[:, 1:T])
    plot!(p1a,  ylabel=L"$-x$ (m)", xlabel=L"$y$ (m)", ylimit=(-(cfg.lane_width_m+1), cfg.lane_width_m+1), xlimit=limits_tuple)
    p1a = add_lane_lines!(p1a, cfg, limits)

    pos_main_filepath = joinpath(folder_name, "LF_passing_scenario_main.pdf")
    savefig(p1a, pos_main_filepath)

    ii = 1
    for t in iter1
        p1b = plot_leadership_filter_measurement_details(num_particles, sg_objs[t], rotated_true_xs[:, 1:T], rotated_x̂s; transform_particle_fn=rotate_particle_state)
        plot!(p1b,  ylabel=L"$-x$ (m)", xlabel=L"$y$ (m)", ylimit=(-(cfg.lane_width_m+1), cfg.lane_width_m+1), xlimit=limits_tuple)
        p1b = add_lane_lines!(p1b, cfg, limits)

        p5, p6 = make_probability_plots(times[1:T], probs[1:T]; t_idx=t)
        plot!(p5, title="")
        plot!(p6, title="")

        pos2_filepath = joinpath(folder_name, "0$(ii)_LF_passing_scenario_positions_detail.pdf")
        prob1_filepath = joinpath(folder_name, "0$(ii)_LF_passing_scenario_probs_P1.pdf")
        prob2_filepath = joinpath(folder_name, "0$(ii)_LF_passing_scenario_probs_P2.pdf")

        savefig(p1b, pos2_filepath)
        savefig(p5, prob1_filepath)
        savefig(p6, prob2_filepath)

        ii += 1
    end

    return true
end

# This generates a gif for the passing scenario, for debugging purposes.
function make_debug_gif(folder_name, filename, cfg, limits, dyn, horizon, times, true_xs, true_us, probs, x̂s, zs, Ts, num_particles, p_transition, num_games)
    T = horizon
    limits_tuple = tuple(limits...)

    rotated_true_xs = rotate_state(dyn, true_xs)
    rotated_zs = rotate_state(dyn, zs)
    rotated_x̂s = rotate_state(dyn, x̂s)
    rotate_particle_state(xs) = rotate_state(dyn, xs)

    # This plot need not be in the loop.
    title="x-y plot of agent positions over time"
    p1a = plot_leadership_filter_positions(dyn, rotated_true_xs[:, 1:T], rotated_x̂s[:, 1:T], rotated_zs[:, 1:T])
    plot!(p1a, title=title,  ylabel=L"$-x$ (m)", xlabel=L"$y$ (m)", ylimit=(-(cfg.lane_width_m+1), cfg.lane_width_m+1), xlimit=limits_tuple)
    p1a = add_lane_lines!(p1a, cfg, limits)

    iter = ProgressBar(2:T)
    anim = @animate for t in iter
        p = @layout [a b; grid(1, 3); e f]

        plot_title = string("LF (", t, "/", T, "), Ts=", Ts, ", Ns=", num_particles, ", p(not transition)=", p_transition, ", #games: ", num_games)
        p1b = plot_leadership_filter_measurement_details(num_particles, sg_objs[t], rotated_true_xs[:, 1:T], rotated_x̂s[:, 1:T]; transform_particle_fn=rotate_particle_state)
        plot!(p1b,  ylabel=L"$-x$ (m)", xlabel=L"$y$ (m)", ylimit=(-(cfg.lane_width_m+1), cfg.lane_width_m+1), xlimit=limits_tuple)
        p1b = add_lane_lines!(p1b, cfg, limits)

        _, p_px, p_py, p_θ, p_v, _, _ = plot_states_and_controls(dyn, times[1:T], true_xs[:, 1:T], true_us)

        # plot 2 - positions
        title1 = "LF est. pos. (x̂/ŷ)"
        plot!(p_px, [times[t], times[t]], [-2, 2], label="", color=:black)
        plot!(p_py, [times[t], times[t]], [-2, 2], label="", color=:black)
        p2 = plot!(p_px, p_py, overlay = true, title=title1)

        # plot 3 - velocities
        title2 = "LF est. heading/velocity (θ̂/v̂)"
        plot!(p_θ, [times[t], times[t]], [-2, 2], label="", color=:black)
        plot!(p_v, [times[t], times[t]], [-2, 2], label="", color=:black)
        p3 = plot!(p_θ, p_v, overlay = true, title=title2)

        # plot 4 - accel. controls
        title5 = "Input acceleration controls (u) over time"
        p4 = plot( xlabel="t (s)", ylabel="accel. (m/s^2)", title=title5)
        plot!(p4, times[1:T], true_us[1][1, 1:T], label=L"\mathcal{A}_1 ω")
        plot!(p4, times[1:T], true_us[2][1, 1:T], label=L"\mathcal{A}_2 ω")
        plot!(p4, times[1:T], true_us[1][2, 1:T], label=L"\mathcal{A}_1 a")
        plot!(p4, times[1:T], true_us[2][2, 1:T], label=L"\mathcal{A}_2 a")
        plot!(p4, [times[t], times[t]], [-1, 1], label="", color=:black)

        # probability plots 5 and 6
        title5 = "Probability over time for P1"
        title6 = "Probability over time for P2"
        p5, p6 = make_probability_plots(times[1:T], probs[1:T]; t_idx=t)
        plot!(p5, title=title5)
        plot!(p6, title=title6)

        plot(p1a, p1b, p2, p3, p4, p5, p6, plot_title=plot_title, layout = p, size=(1260, 1080))
    end


    # Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
    previous_GKSwstype = get(ENV, "GKSwstype", "")
    ENV["GKSwstype"] = "100"

    println("giffying...")
    filename = joinpath(folder_name, filename)
    gif(anim, filename, fps=10)
    println("done")

    # Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
    ENV["GKSwstype"] = previous_GKSwstype
end
