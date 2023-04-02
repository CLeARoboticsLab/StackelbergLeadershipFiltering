include("RunSILQGamesOnLQExample.jl")

using Plots
gr()

N = Int(sg_obj.num_iterations[1]+1)
anim = @animate for k in 1:N
    println(k)
    p = @layout [a b; c d; e f]

    xns = sg_obj.xks[1, k, :, :]
    un1s = sg_obj.uks[1][1, k, :, :]
    un2s = sg_obj.uks[2][1, k, :, :]

    plot(legend=:outertopright)
    plot!(q1, xns[1, :], xns[3, :], label="")
    plot!(q1, xns[5, :], xns[7, :], label="")

    # q1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright, title="Iterative LQR")
    p1 = scatter!([x₁[1]], [x₁[3]], color="blue", label="")
    p1 = scatter!([x₁[5]], [x₁[7]], color="red", label="")

    p2 = plot(times, xns[1,:], label="P1 px", legend=:outertopright)
    plot!(times, xns[3,:], label="P1 py")
    plot!(times, xns[5,:], label="P2 px", legend=:outertopright)
    plot!(times, xns[7,:], label="P2 py")

    p3 = plot(times, xns[2,:], label="vel1 x", legend=:outertopright)
    plot!(times, xns[4,:], label="vel1 y")
    plot!(times, xns[6,:], label="vel2 x")
    plot!(times, xns[8,:], label="vel2 y")

    p4 = plot(times, un1s[1, :], label="P1 accel x", legend=:outertopright)
    plot!(times, un1s[2, :], label="P1 accel y")
    plot!(times, un2s[1, :], label="P2 accel x", legend=:outertopright)
    plot!(times, un2s[2, :], label="P2 accel y")

    plot(p1, p2, p3, p4, layout = p)
end
println("giffying...")
gif(anim, "silqgames_animation.gif", fps = 10)
println("done")