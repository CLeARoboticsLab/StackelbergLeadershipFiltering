# include("RunSILQGamesOnLQExample.jl")
# include("RunSILQGamesOnQuadraticNonlinearGame.jl")
include("RunSILQGamesWithNonLQExample.jl")

using Plots
using ProgressBars
gr()

# These scripts should require one run of SILQGames.
@assert sg_obj.num_runs == 1
N = sg_obj.num_iterations[1]

iter = ProgressBar(1:N)
anim = @animate for k in iter
    p = @layout [a b c; d e f; g h i]

    xns = sg_obj.xks[1, k, :, :]
    un1s = sg_obj.uks[1][1, k, :, :]
    un2s = sg_obj.uks[2][1, k, :, :]

    p1, p2, p3, p4, p5, p6, p7 = plot_states_and_controls(dyn, times, xns, [un1s, un2s])

    # Plot convergence.
    conv_x = cumsum(ones(num_iters)) .- 1
    r1 = plot(conv_x, conv_metrics[1, 1:num_iters], title="||k||^2", label="p1", yaxis=:log)
    plot!(r1, conv_x, conv_metrics[2, 1:num_iters], label="p2", yaxis=:log)
    # plot!(r1, [k, k], [0, max(conv_metrics[1, 1], conv_metrics[2, 1])], label="", color=:black)

    r2 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label="p1", yaxis=:log)
    plot!(r2, conv_x, evaluated_costs[2, 1:num_iters], label="p2", yaxis=:log)
    # plot!(r2, [k, k], [0, max(evaluated_costs[1, 1], evaluated_costs[2, 1])], label="", color=:black)

    plot(p1, p2, p3, p4, p5, p6, p7, r1, r2, layout = p)
end

# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
gif(anim, "silqgames_animation.gif", fps = 20)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
