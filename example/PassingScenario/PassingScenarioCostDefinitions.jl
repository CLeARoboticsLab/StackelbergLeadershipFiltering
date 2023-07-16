using StackelbergControlHypothesesFiltering
using LinearAlgebra: norm, Diagonal, I
using Random: MersenneTwister
using Distributions: Bernoulli, MvNormal

include("CreatePassingScenarioGame.jl")
include("GroundTruthUtils.jl")
include("PassingScenarioConfig.jl")

# Define game and timing related configuration.
num_players = 2

T = 101
t₀ = 0.0
dt = 0.05
horizon = T * dt
times = dt * cumsum(ones(2*T)) .- dt

# Get the configuration.
cfg = PassingScenarioConfig(collision_radius_m=0.0)
                            # max_heading_deviation=pi/6)

# Defined the dynamics of the game.
dyn = create_passing_scenario_dynamics(num_players, dt)
si = dyn.sys_info

# Define the starting and goal state.
v_init = 10.
rlb_x = get_right_lane_boundary_x(cfg)
x₁ = [rlb_x/2; 10.; pi/2; v_init; rlb_x/4.; 0.; pi/2; 2*v_init]

p1_goal = vcat([x₁[1]; 40.; pi/2; 0.], zeros(4))
p2_goal = vcat(zeros(4),                [x₁[5]; 120.; pi/2; 2*v_init])

# Define the costs for the agents.
num_subcosts = 14
weights_p1 = ones(num_subcosts)
weights_p2 = ones(num_subcosts)

# Adjust goal tracking weights.
weights_p1[1] = 1.
weights_p2[2] = 1.
costs = create_passing_scenario_costs(cfg, si, weights_p1, weights_p2, p1_goal, p2_goal)

# Generate a ground truth trajectory on which to run the leadership filter.
gt_threshold=1e-3
gt_max_iters=120
gt_step_size=1e-2
gt_verbose=true
gt_num_runs=1
sg_obj = initialize_silq_games_object(gt_num_runs, T, dyn, costs;
                                      threshold=gt_threshold, max_iters=gt_max_iters, step_size=gt_step_size, verbose=gt_verbose)

# An initial control estimate.
gt_leader_idx = 1
us_1 = [zeros(udim(si, 1), T), zeros(udim(si, 1), T)]
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = generate_gt_from_silqgames(sg_obj, gt_leader_idx, times, x₁, us_1)
plot_silqgames_gt(dyn, times[1:T], xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs)



# # Run the leadership filter.

# Initial condition chosen randomly. Ensure both have relatively low speed.
pos_unc = 1e-2
θ_inc = 1e-2
vel_unc = 1e-3
P₁ = Diagonal([pos_unc, pos_unc, θ_inc, vel_unc, pos_unc, pos_unc, θ_inc, vel_unc])

# Process noise uncertainty
Q = 1e0 * Diagonal([1e-2, 1e-2, 1e-3, 1e-4, 1e-2, 1e-2, 1e-3, 1e-4])


# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 0.05 * I
zs = zeros(xdim(dyn), T)
Ts = 20
num_games = 1
num_particles = 100

p_transition = 0.98
p_init = 0.3

discrete_state_transition, state_trans_P = generate_discrete_state_transition(p_transition, p_transition)
s_init_distrib = Bernoulli(p_init)

process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q)


threshold = 1e-2
max_iters = 50
step_size = 1e-2

# Augment the remaining states so we have T+Ts-1 of them.
true_xs = hcat(xs_k, zeros(xdim(dyn), Ts-1))
true_us = [hcat(us_k[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

# Fill in z as noisy state measurements.
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(true_xs[:, tt], R))
end

x̂s, P̂s, probs, pf, sg_objs = leadership_filter(dyn, costs, t₀, times,
                           T,         # simulation horizon
                           Ts,        # horizon over which the stackelberg game should be played,
                           num_games, # number of stackelberg games played for measurement
                           x₁,        # initial state at the beginning of simulation
                           P₁,        # initial covariance at the beginning of simulation
                           us_k,      # the control inputs that the actor takes
                           zs,        # the measurements
                           R,
                           process_noise_distribution,
                           s_init_distrib,
                           discrete_state_transition;
                           threshold=threshold,
                           rng,
                           max_iters=max_iters,
                           step_size=step_size,
                           Ns=num_particles,
                           verbose=false)

using Dates
using Plots
using Printf
using ProgressBars
gr()

# N = Int(sg_obj.num_iterations[1]+1)
iter = ProgressBar(2:T)
anim = @animate for t in iter
    p = @layout [a; b c; d e; f]

    plot_title = string("LF (", t, "/", T, ") on Stack(L=P", gt_leader_idx, "), Ts=", Ts, ", Ns=", num_particles, ", p(transition)=", p_transition, ", #games: ", num_games)
    println(plot_title)
    title1="x-y plot of agent positions over time"
    p1 = plot(title=title1, legend=:outertopright, ylabel="-x (m)", xlabel="y (m)", ylimit=(-(cfg.lane_width_m+1), cfg.lane_width_m+1), xlimit=(-5., 75.))
    plot!(p1, true_xs[2, 1:T], -true_xs[1, 1:T], label="P1 true pos")
    plot!(p1, true_xs[6, 1:T], -true_xs[5, 1:T], label="P2 true pos")

    plot!(p1, zs[2, 1:T], -zs[1, 1:T], label="P1 meas pos", color=:blue, linewidth=0.15)
    plot!(p1, zs[6, 1:T], -zs[5, 1:T], label="P2 meas pos", color=:red, linewidth=0.15)

    plot!(p1, x̂s[2, 1:T], -x̂s[1, 1:T], label="P1 est. pos", color=:blue, linewidth=0.25)
    plot!(p1, x̂s[6, 1:T], -x̂s[5, 1:T], label="P2 est. pos", color=:red, linewidth=0.25)

    p1 = scatter!([x₁[2]], [-x₁[1]], color="blue", label="start P1")
    p1 = scatter!([x₁[6]], [-x₁[5]], color="red", label="start P2")

    # plot 2
    title2 = "LF estimated states (x̂) over time"
    p2 = plot(legend=:outertopright, xlabel="t (s)", ylabel="pos (m)", title=title2)
    plot!(p2, times[1:T], x̂s[1,1:T], label="P1 px")
    plot!(p2, times[1:T], x̂s[2,1:T], label="P1 py")
    plot!(p2, times[1:T], x̂s[5,1:T], label="P2 px")
    plot!(p2, times[1:T], x̂s[6,1:T], label="P2 py")
    plot!(p2, [times[t], times[t]], [-1, 2], label="", color=:black)

    # plot 3
    title3 = "LF estimated heading (x̂) over time"
    p3 = plot(legend=:outertopright, xlabel="t (s)", ylabel="θ (rad)", title=title3)
    plot!(p3, times[1:T], x̂s[3,1:T], label="P1 θ")
    plot!(p3, times[1:T], x̂s[7,1:T], label="P2 θ")
    plot!(p3, [times[t], times[t]], [-pi, pi], label="", color=:black)

    # plot 3
    title3 = "LF estimated velocity (x̂) over time"
    p4 = plot(legend=:outertopright, xlabel="t (s)", ylabel="vel (m/2)", title=title3)
    plot!(p4, times[1:T], x̂s[4,1:T], label="P1 v")
    plot!(p4, times[1:T], x̂s[8,1:T], label="P2 v")
    plot!(p4, [times[t], times[t]], [0, cfg.speed_limit_mps], label="", color=:black)


    # Add particles
    num_iters = [0, 0]
    for n in 1:num_particles

        num_iter = sg_objs[t].num_iterations[n]

        x1_idx = 1
        y1_idx = 2
        x2_idx = 5
        y2_idx = 6

        xks = sg_objs[t].xks[n, num_iter, :, :]

        # TODO(hamzah) - change color based on which agent is leader
        scatter!(p1, xks[y1_idx, :], -xks[x1_idx, :], color=:black, markersize=0.5, label="")

        color = (sg_objs[t].leader_idxs[n] == 1) ? :blue : :red
        scatter!(p1, [xks[y1_idx, 2]], -[xks[x1_idx, 2]], color=color, markersize=3., label="")

        scatter!(p1, xks[y2_idx, :], -xks[x2_idx, :], color=:black, markersize=0.5, label="")
        scatter!(p1, [xks[y2_idx, 2]], -[xks[x2_idx, 2]], color=color, markersize=3., label="")
    end

    # plot 4
    title5 = "Input acceleration controls (u) over time"
    p5 = plot(legend=:outertopright, xlabel="t (s)", ylabel="accel. (m/s^2)", title=title5)
    plot!(p5, times[1:T], us_k[1][1, 1:T], label="P1 ω")
    plot!(p5, times[1:T], us_k[2][1, 1:T], label="P2 ω")
    plot!(p5, times[1:T], us_k[1][2, 1:T], label="P1 a")
    plot!(p5, times[1:T], us_k[2][2, 1:T], label="P2 a")
    plot!(p5, [times[t], times[t]], [-1, 1], label="", color=:black)

    # probability plot - plot 5
    title6 = "Probability over time"
    p6 = plot(xlabel="t (s)", ylabel="prob. leadership", ylimit=(-0.1, 1.1), label="", legend=:outertopright, title=title6)
    plot!(p6, times[1:T], probs[1:T], label="p(L=P1)")
    plot!(p6, [times[t], times[t]], [0, 1], label="", color=:black)

    plot(p1, p2, p3, p4, p5, p6, plot_title=plot_title, layout = p, size=(1260, 1080))
end

# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
filename = string("passing_scenario_",string(Dates.now()),".gif")
gif(anim, filename, fps=10)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
