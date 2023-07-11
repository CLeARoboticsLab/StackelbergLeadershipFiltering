using StackelbergControlHypothesesFiltering
using LinearAlgebra: norm, Diagonal, I
using Random: MersenneTwister
using Distributions: Bernoulli, MvNormal

num_players = 2

T = 101
t0 = 0.0
dt = 0.05
horizon = T * dt
times = dt * cumsum(ones(2*T)) .- dt

dyn = UnicycleDynamics(num_players, dt)
si = dyn.sys_info


lane_width_m = 2.3
cl_x = 0.0
llb_x = -lane_width_m
rlb_x = lane_width_m
speed_limit_mps = 35

v_init = 10.
p1_goal = vcat([rlb_x/2; 40.; pi/2; 0.], zeros(4))
p2_goal = vcat(zeros(4),                [rlb_x/2; 60.; pi/2; 0.])


# Passing Scenario cost definition
# 1. distance to goal (no control cost)

# TODO(hamzah) - should this be replaced with a tracking trajectory cost?

const_1 = 10.

Q1 = zeros(8, 8)
Q1[1, 1] = const_1 * 1.
Q1[2, 2] = const_1 * 1.
Q1[3, 3] = const_1 * 1.
Q1[4, 4] = const_1 * 1.
q_cost1 = QuadraticCost(Q1)
add_control_cost!(q_cost1, 1, zeros(udim(si, 1), udim(si, 1)))
add_control_cost!(q_cost1, 2, zeros(udim(si, 2), udim(si, 2)))
c1a = QuadraticCostWithOffset(q_cost1, p1_goal)

Q2 = zeros(8, 8)
Q2[5, 5] = const_1 * 1.
Q2[6, 6] = const_1 * 1.
Q2[7, 7] = const_1 * 1.
Q2[8, 8] = const_1 * 1.
q_cost2 = QuadraticCost(Q2)
add_control_cost!(q_cost2, 1, zeros(udim(si, 1), udim(si, 1)))
add_control_cost!(q_cost2, 2, zeros(udim(si, 2), udim(si, 2)))
c2a = QuadraticCostWithOffset(q_cost2, p2_goal)

# 2. avoid collisions
NORM_ORDER = 2
f(si, x, us, t) = begin
   return -log(norm([1 1 0 0 -1 -1 0 0] * x, NORM_ORDER))
end

c1b = PlayerCost(f, si)
c2b = PlayerCost(f, si)

# 3. enforce speed limit and turning limit
c1c_i = AbsoluteLogBarrierCost(4, speed_limit_mps, false)
c1c_ii = AbsoluteLogBarrierCost(4, -speed_limit_mps, true)

θ₀ = pi/2 # pointed in y-direction
Δθ = pi/3 # max difference from θ₀
c1c_iii = AbsoluteLogBarrierCost(3, θ₀+Δθ, false)
c1c_iv = AbsoluteLogBarrierCost(3, θ₀-Δθ, true)

c2c_i = AbsoluteLogBarrierCost(8, speed_limit_mps, false)
c2c_ii = AbsoluteLogBarrierCost(8, -speed_limit_mps, true)
c2c_iii = AbsoluteLogBarrierCost(7, θ₀+Δθ, false)
c2c_iv = AbsoluteLogBarrierCost(7, θ₀-Δθ, true)

# 4, 5. minimize and bound control effort
MAX_ROTVEL = 2. # arbitrarily chosen
MAX_ACCEL = 9. # this corresponds to 0 to 60 in 2.5s, which is close to the record

c1de = QuadraticCost(zeros(8, 8), zeros(8), 0.)
R11 = [1. 0; 0 1.]
add_control_cost!(c1de, 1, R11)
add_control_cost!(c1de, 2, zeros(udim(si, 2), udim(si, 2)))

c1de_i = AbsoluteLogBarrierControlCost(1, [1.; 0.], MAX_ROTVEL, false)
c1de_ii = AbsoluteLogBarrierControlCost(1, [1.; 0.], -MAX_ROTVEL, true)
c1de_iii = AbsoluteLogBarrierControlCost(1, [0.; 1.], MAX_ACCEL, false)
c1de_iv = AbsoluteLogBarrierControlCost(1, [0.; 1.], -MAX_ACCEL, true)

c2de = QuadraticCost(zeros(8, 8))
R22 = [1. 0; 0 1.]
add_control_cost!(c2de, 2, R22)
add_control_cost!(c2de, 1, zeros(udim(si, 1), udim(si, 1)))

c2de_i = AbsoluteLogBarrierControlCost(2, [1.; 0.], MAX_ROTVEL, false)
c2de_ii = AbsoluteLogBarrierControlCost(2, [1.; 0.], -MAX_ROTVEL, true)
c2de_iii = AbsoluteLogBarrierControlCost(2, [0.; 1.], MAX_ACCEL, false)
c2de_iv = AbsoluteLogBarrierControlCost(2, [0.; 1.], -MAX_ACCEL, true)

# 6. log barriers on the x dimension ensure that the vehicles don't exit the road
# TODO(hamzah) - remove assumption of straight road
c1f_i = AbsoluteLogBarrierCost(1, rlb_x, false)
c1f_ii = AbsoluteLogBarrierCost(1, llb_x, true)

c2f_i = AbsoluteLogBarrierCost(5, rlb_x, false)
c2f_ii = AbsoluteLogBarrierCost(5, llb_x, true)

# 7. Add a bump in cost for crossing the centerline.
c1g = GaussianCost([1], [cl_x], ones(1, 1))
c2g = GaussianCost([5], [cl_x], ones(1, 1))


# Combine them together.
function combine_cost_funcs(funcs, weights)
    @assert size(funcs) == size(weights)

    # Create a weighted cost function ready for use with autodiff.
    g(si, x, us, t) = begin
        f_eval = [f(si, x, us, t) for f in funcs]
        return weights' * f_eval
    end

    return g
end

costs_p1 = [c1a,
            c1b,
            c1c_i,
            c1c_ii,
            c1c_iii,
            c1c_iv,
            c1de,
            c1de_i,
            c1de_ii,
            c1de_iii,
            c1de_iv,
            c1f_i,
            c1f_ii,
            c1g
            ]
num_costs_p1 = length(costs_p1)
weights_p1 = ones(num_costs_p1)

costs_p2 = [c2a,
            c2b,
            c2c_i,
            c2c_ii,
            c2c_iii,
            c2c_iv,
            c2de,
            c2de_i,
            c2de_ii,
            c2de_iii,
            c2de_iv,
            c2f_i,
            c2f_ii,
            c2g
            ]
num_costs_p2 = length(costs_p2)
weights_p2 = ones(num_costs_p2)

g1 = combine_cost_funcs(get_as_function.(costs_p1), weights_p1)
sum_cost_p1 = PlayerCost(g1, si)

g2 = combine_cost_funcs(get_as_function.(costs_p2), weights_p2)
sum_cost_p2 = PlayerCost(g2, si)


costs = [sum_cost_p1, sum_cost_p2]



# config variables
gt_threshold=1e-3
gt_max_iters=280
gt_step_size=1e-2
gt_verbose=true

leader_idx = 1
x₁ = [rlb_x/2; 10.; pi/2; v_init; rlb_x/2.2; 0.; pi/2; 2 * v_init]
us_1 = [zeros(udim(si, 1), T), zeros(udim(si, 1), T)]

num_runs=1
sg_obj = initialize_silq_games_object(num_runs, T, dyn, costs;
                                      threshold=gt_threshold, max_iters=gt_max_iters, step_size=gt_step_size, verbose=gt_verbose)
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times, x₁, us_1)

println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals, " sum: ", sum(final_cost_totals))

using Plots

# PLOTS A: Plot states/controls.
l = @layout [
    a{0.3h}; [grid(2,3)]
]

# q = @layout [a b; c d ;e f; g h]
pos_plot, p2, p3, p4, p5, p6, p7 = plot_states_and_controls(dyn, times, xs_k, us_k)
plot(pos_plot, p2, p3, p4, p5, p6, p7, layout = l)


# PLOTS B: Plot convergence metrics/costs separately.
m = @layout [a; b]

conv_x = cumsum(ones(num_iters)) .- 1
title8 = "conv. (|⋅|∞)"
q8 = plot(title=title8, yaxis=:log, legend=:outertopright)
plot!(conv_x, conv_metrics[1, 1:num_iters], label="p1")
plot!(conv_x, conv_metrics[2, 1:num_iters], label="p2")

conv_sum = conv_metrics[1, 1:num_iters] + conv_metrics[2, 1:num_iters]
plot!(conv_x, conv_sum, label="total")

title9 = "evaluated costs"
q9 = plot(title=title9, yaxis=:log, legend=:outertopright)
plot!(conv_x, evaluated_costs[1, 1:num_iters], label="p1")
plot!(conv_x, evaluated_costs[2, 1:num_iters], label="p2")

cost_sum = evaluated_costs[1, 1:num_iters] + evaluated_costs[2, 1:num_iters]
plot!(conv_x, cost_sum, label="total")

plot(q8, q9, layout = m)





# # Run the leadership filter.

t0 = 0.
leader_idx = 1
# Initial condition chosen randomly. Ensure both have relatively low speed.
pos_unc = 1e-2
θ_inc = 1e-2
vel_unc = 1e-3
P₁ = Diagonal([pos_unc, pos_unc, θ_inc, vel_unc, pos_unc, pos_unc, θ_inc, vel_unc])

# Process noise uncertainty
Q = 1e-2 * Diagonal([1e-2, 1e-2, 1e-3, 1e-4, 1e-2, 1e-2, 1e-3, 1e-4])


# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 0.1 * I
zs = zeros(xdim(dyn), T)
Ts = 50
num_games = 1
num_particles = 100

p_transition = 0.98
p_init = 0.3

discrete_state_transition, state_trans_P = generate_discrete_state_transition(p_transition, p_transition)
s_init_distrib = Bernoulli(p_init)

process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q)


threshold = 1e-2
max_iters = 25
step_size = 1e-2

# Augment the remaining states so we have T+Ts-1 of them.
true_xs = hcat(xs_k, zeros(xdim(dyn), Ts-1))
true_us = [hcat(us_k[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

# Fill in z as noisy state measurements.
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(true_xs[:, tt], R))
end

x̂s, P̂s, probs, pf, sg_objs = leadership_filter(dyn, costs, t0, times,
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

    plot_title = string("LF (", t, "/", T, ") on Stack(L=P", leader_idx, "), Ts=", Ts, ", Ns=", num_particles, ", p(transition)=", p_transition, ", #games: ", num_games)
    println(plot_title)
    title1="x-y plot of agent positions over time"
    p1 = plot(title=title1, legend=:outertopright, ylabel="-x (m)", xlabel="y (m)", ylimit=(-(lane_width_m+1), lane_width_m+1), xlimit=(-5., 75.))
    plot!(p1, true_xs[2, 1:T], -true_xs[1, 1:T], label="P1 true pos")
    plot!(p1, true_xs[6, 1:T], -true_xs[5, 1:T], label="P2 true pos")

    plot!(p1, zs[2, 1:T], -zs[1, 1:T], label="P1 meas pos", color=:blue, linewidth=0.15)
    plot!(p1, zs[6, 1:T], -zs[5, 1:T], label="P2 meas pos", color=:red, linewidth=0.15)

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
    plot!(p4, [times[t], times[t]], [-1, 1], label="", color=:black)


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
    plot!(p6, times[1:T], (leader_idx%2) * ones(T), label="truth")
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
