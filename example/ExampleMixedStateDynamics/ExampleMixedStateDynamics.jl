# Mixed state dynamics with one discrete state.

using StackelbergControlHypothesesFiltering

using Distributions
using LinearAlgebra
using Plots
using Random
using Reel

rng = Random.GLOBAL_RNG;

goal_1 = 10.
goal_2 = -10.

function get_goal_dynamics(goal_pos::Float64, Δt::Float64)
    A = [1  0;
         Δt 1]
    Bs = [[1; 0][:, :]]
    D = [1 0; 0 1] # 1D position noise
    return LinearDynamics(A, Bs, D)
end

# time step
Δt = 1.

# Define noise.
v_bar = zeros(2)
Q = diagm([0.01, 0.5])

x0_bar = [0; 1]
P0 = diagm([0.1, 1])

ϵ_bar = zeros(1)
R = 0.5 * ones(1, 1)

# dynamics definitions
dyn_probs = [0.000001, 0.999999]
dyn_right = get_goal_dynamics(goal_1, Δt)
dyn_left = get_goal_dynamics(goal_2, Δt)
dyns = [dyn_right, dyn_left]

num_states = xdim(dyn_left)

# generate true rollout
num_steps = 50
x_trues = zeros(num_states, num_steps)
x_trues[:, 1] = x0_bar
for i in 2:num_steps
    # Randomly select which dynamics to use for this cycle.
    dyn_idx = findfirst(cumsum(dyn_probs) .> rand())
    # println(i, " - ", dyn_idx)
    dyn = dyns[dyn_idx]
    t = Δt * (i-1)

    # generate control
    goal = (dyn_idx == 1) ? goal_1 : goal_2
    pos_diff = goal - x_trues[2, i-1]

    vel_diff = pos_diff * Δt^(-2) - x_trues[1, i-1]

    vel_ctrl_val = sign(vel_diff) * min(abs(vel_diff), 1.0)
    vel_ctrl = [zeros(udim(dyn, 1))]
    vel_ctrl[1][1] = vel_ctrl_val

    no_process_noise = v_bar
    x_trues[:, i] = propagate_dynamics(dyn, t, x_trues[:, i-1], vel_ctrl, no_process_noise)
end

println(x_trues)

# For the true rollout, generate measurements to plot for now.
# TODO(hamzah): Replace this with the particles.
num_meas_points = 100
meas_pts = zeros(1, num_meas_points, num_steps)
for i in 1:num_steps
    # println(i, " - ", x_trues[2, i])
    meas_pts[:, :, i] = rand(rng, Normal(x_trues[2, i], R[1, 1]), num_meas_points)[:, :]
end

# multi-state particle filter stuff



# plotting

times = [Δt * (i-1) for i in 1:num_steps]
# Xs = ones(2, num_steps)

# make plots at each time step
plt = plot(times, x_trues[1:2, :]')
# plt = scatter!(plt, [times[i] for i in 1:num_steps], [meas_pts[1, :, i] for i in 1:num_steps], color=:black, markersize=0.75)



# plots = []

# for i in 1:num_steps
#     # m = mean(b)
#     # u = [-m[1], -m[2]] # Control law - try to orbit the origin
#     # x = f(x, u, rng)
#     # y = h(x, rng)
#     # b = update(filter, b, u, y)

#     # plt = scatter!(plt, [p[1] for p in particles(b)], [p[2] for p in particles(b)], color=:black, markersize=0.1, label="")
#     plt = scatter([x_trues[1, i]], [x_trues[2, i]], color=:blue, xlim=(-150,150), ylim=(-100,100), label="")
#     push!(plots, plt)
# end

# # Make a gif
# frames = Frames(MIME("image/png"), fps=10)
# for plt in plots
#     print(".")
#     push!(frames, plt)
# end
# print("writing")
# write("output.gif", frames)
