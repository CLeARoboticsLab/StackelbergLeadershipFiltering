# include("RunSILQGamesOnLQExample.jl")
# include("RunSILQGamesOnQuadraticNonlinearGame.jl")
# include("RunSILQGamesOnNonquadraticLinearGame.jl")
include("RunSILQGamesOnNonLQGame.jl")

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
    r1 = plot(conv_x, conv_metrics[1, 1:num_iters], title="conv.", label="p1", yaxis=:log)
    plot!(r1, conv_x, conv_metrics[2, 1:num_iters], label="p2", yaxis=:log)
    plot!(r1, [k, k], [minimum(conv_metrics[:, 1:num_iters]), maximum(conv_metrics[:, 1:num_iters])], label="", color=:black, yaxis=:log)

    # r2 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label="p1", yaxis=:log)
    # plot!(r2, conv_x, evaluated_costs[2, 1:num_iters], label="p2", yaxis=:log)
    # plot!(r2, [k, k], [minimum(evaluated_costs[:, 1:num_iters]), maximum(evaluated_costs[:, 1:num_iters])], label="", color=:black, yaxis=:log)

    # Shift the cost to ensure they are positive.
    costs_1 = evaluated_costs[1, 1:num_iters] .+ (abs(minimum(evaluated_costs[1, 1:num_iters])) + 1e-8)
    costs_2 = evaluated_costs[2, 1:num_iters] .+ (abs(minimum(evaluated_costs[2, 1:num_iters])) + 1e-8)

    q6 = plot(conv_x, costs_1, title="evaluated costs", label="p1", yaxis=:log)
    plot!(q6, conv_x, costs_2, label="p2", yaxis=:log)
    plot!(q6, [k, k], [minimum(costs_1), maximum(costs_2)], label="", color=:black, yaxis=:log)

    plot(p1, p2, p3, p4, p5, p6, p7, r1, q6, layout = p)
end

# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
gif(anim, "silqgames_animation.gif", fps = 50)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
