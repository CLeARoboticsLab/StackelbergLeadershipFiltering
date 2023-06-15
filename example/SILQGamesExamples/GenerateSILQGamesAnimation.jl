# include("RunSILQGamesOnLQExample.jl")
include("RunSILQGamesOnQuadraticNonlinearGame.jl")

using Plots
using ProgressBars
gr()

# These scripts should require one run of SILQGames.
@assert sg_obj.num_runs == 1
N = sg_obj.num_iterations[1]

iter = ProgressBar(1:N)
anim = @animate for k in iter
    p = @layout [a b; c d ; e f; g h]

    xns = sg_obj.xks[1, k, :, :]
    un1s = sg_obj.uks[1][1, k, :, :]
    un2s = sg_obj.uks[2][1, k, :, :]

    p1, p2, p3, p4, p5, p6, p7 = plot_states_and_controls(dyn, times, xns, [un1s, un2s])

    plot(p1, p2, p3, p4, p5, p6, p7, layout = p)
end

# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
gif(anim, "silqgames_animation.gif", fps = 50)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
