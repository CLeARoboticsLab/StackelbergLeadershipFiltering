include("RunLeadershipFilterOnKnownStackelbergGame.jl")

using Dates
using Plots
using Printf
using ProgressBars
gr()

# N = Int(sg_obj.num_iterations[1]+1)
iter = ProgressBar(2:T)
anim = @animate for t in iter
    p = @layout [a; b]

    title = string("LF (", t, ") on Stack(", leader_idx, "), Ts=", Ts,"\n, Ns=", num_particles)
    p1 = plot(legend=:outertopright, ylabel="y (m)", xlabel="x (m)", title=title)
    plot!(p1, true_xs[1, 1:T], true_xs[3, 1:T], label="P1 pos", ylimit=(-2.0, 2.0), xlimit=(-2.0, 2.0))
    plot!(p1, true_xs[5, 1:T], true_xs[7, 1:T], label="P2 pos")

    plot!(p1, zs[1, 1:T], zs[3, 1:T], label="P1 meas pos", color=:blue, linewidth=0.15)
    plot!(p1, zs[5, 1:T], zs[7, 1:T], label="P2 meas pos", color=:red, linewidth=0.15)

    # p1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright, title="Leadership Filter")
    p1 = scatter!([x₁[1]], [x₁[3]], color="blue", label="start P1")
    p1 = scatter!([x₁[5]], [x₁[7]], color="red", label="start P2")

    num_iters = [0, 0]
    for n in 1:num_particles

        num_iter_1 = sg_objs[t].num_iterations[n]
        num_iter_2 = sg_objs[t].num_iterations[n]

        # println("particle n thinks leader is: ", n)
        # println("num iters 1, 2: ", sg_objs[t].num_iterations, " ", sg_objs[t].num_iterations[n])
        # println("num iters 1, 2: ", sg_objs[t].num_iterations, " ", sg_objs[t].num_iterations[n])

        x1_idx = 1
        y1_idx = 3
        x2_idx = 5
        y2_idx = 7

        # TODO(hamzah) - change color based on which agent is leader
        scatter!(p1, sg_objs[t].xks[n, num_iter_1, x1_idx, :], sg_objs[t].xks[n, num_iter_1, y1_idx, :], color=:black, markersize=0.5, label="")

        color = (sg_objs[t].leader_idxs[n] == 1) ? :blue : :red
        scatter!([sg_objs[t].xks[n, num_iter_1, x1_idx, 2]], [sg_objs[t].xks[n, num_iter_1, y1_idx, 2]], color=color, markersize=3., label="")

        scatter!(p1, sg_objs[t].xks[n, num_iter_2, x2_idx, :], sg_objs[t].xks[n, num_iter_2, y2_idx, :], color=:black, markersize=0.5, label="")
        scatter!([sg_objs[t].xks[n, num_iter_1, x2_idx, 2]], [sg_objs[t].xks[n, num_iter_1, y2_idx, 2]], color=color, markersize=3., label="")
    end

    # p2 = plot(times, xns[1,:], label="P1 px", legend=:outertopright)
    # plot!(times, xns[3,:], label="P1 py")
    # plot!(times, xns[5,:], label="P2 px", legend=:outertopright)
    # plot!(times, xns[7,:], label="P2 py")

    # p3 = plot(times, xns[2,:], label="vel1 x", legend=:outertopright)
    # plot!(times, xns[4,:], label="vel1 y")
    # plot!(times, xns[6,:], label="vel2 x")
    # plot!(times, xns[8,:], label="vel2 y")

    # p4 = plot(times, un1s[1, :], label="P1 accel x", legend=:outertopright)
    # plot!(times, un1s[2, :], label="P1 accel y")
    # plot!(times, un2s[1, :], label="P2 accel x", legend=:outertopright)
    # plot!(times, un2s[2, :], label="P2 accel y")

    # probability plot
    p5 = plot(times[1:T], probs[1:T], xlabel="t (s)", ylabel="prob. leadership", ylimit=(-0.1, 1.1))
    scatter!([times[t]], [probs[t]], color=:black, markersize=3.)

    plot(p1, p5, layout = p)
end
println("giffying...")
filename = string("lq_leadfilt_animation_",string(Dates.now()),".gif")
gif(anim, filename, fps = 10)
println("done")
