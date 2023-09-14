# A file for utilities for generating the ground truth that is passed into the leadership filter.
# There are multiple ways to do this, but for now we use SILQGames to generate it.

using StackelbergControlHypothesesFiltering

using JLD
using Plots

include("MergingScenarioConfig.jl")
include("PassingScenarioConfig.jl")

function get_passing_trajectory_scenario_101(cfg::PassingScenarioConfig, x₁)

    rlb_x = get_right_lane_boundary_x(cfg)
    v_init = 10

    # x₁ = [rlb_x/2; 10.; pi/2; v_init; rlb_x/1.5; 0.; pi/2; v_init]

    w1 = zeros(2, 101) # Agent 1 keeps going in straight line

    # Agent 2 does a passing maneuver.
    w2 = vcat(hcat( ones(1, 8),
                   -ones(1, 8),
                   zeros(1, 60),
                   -ones(1, 8),
                    ones(1, 8),
                   zeros(1, 9)), 
              hcat( 5  *ones(1, 10),
                    8.7*ones(1, 15),
                       zeros(1, 49),
                   -8.7*ones(1, 17),
                   -5  *ones(1, 10)))

    ws = [w1, w2]

    xs = unroll_raw_controls(dyn, times[1:T], ws, x₁)
    return xs, ws
end

function get_passing_trajectory_scenario_151(cfg::PassingScenarioConfig, x₁; T=151)
    rlb_x = get_right_lane_boundary_x(cfg)
    v_init = 10

    # x₁ = [rlb_x/2; 10.; pi/2; v_init; rlb_x/1.5; 0.; pi/2; v_init]

    w1 = zeros(2, 151) # Agent 1 keeps going in straight line

    # Agent 2 does a passing maneuver after a short follow.
    w2 = vcat(hcat(zeros(1, 50),
                    ones(1, 8),
                   -ones(1, 8),
                   zeros(1, 60),
                   -ones(1, 8),
                    ones(1, 8),
                   zeros(1, 9)), 
              hcat(    zeros(1, 50),
                    5  *ones(1, 10),
                    8.7*ones(1, 15),
                       zeros(1, 49),
                   -8.7*ones(1, 17),
                   -5  *ones(1, 10)))

    ws = [w1[:, 1:T], w2[:, 1:T]]

    xs = unroll_raw_controls(dyn, times[1:T], ws, x₁)
    return xs, ws
end

function get_merging_trajectory_p1_straight_31(cfg::MergingScenarioConfig, x₁)
    w1 = vcat(hcat(zeros(1, 31),
                  # -0.2*ones(1, 8),
                  #  zeros(1, 10),
                  #  0.2*ones(1, 8),
                  #  zeros(1, 9),
                  #  zeros(1, 16),
                  #  zeros(1, 20)
                ),
              hcat( #4  *ones(1, 10),
              #       7.7*ones(1, 15),
                       # zeros(1, 49),
                       zeros(1, 31)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

     w2 = vcat(hcat(zeros(1, 31),
                  # -0.2*ones(1, 8),
                  #  zeros(1, 10),
                  #  0.2*ones(1, 8),
                  #  zeros(1, 9),
                  #  zeros(1, 16),
                  #  zeros(1, 20)
                ),
              hcat( #4  *ones(1, 10),
              #       7.7*ones(1, 15),
                       # zeros(1, 49),
                       zeros(1, 31)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

    # # Agent 2 does a passing maneuver.
    # w2 = vcat(hcat(zeros(1, 40),
    #                0.2*ones(1, 8),
    #                zeros(1, 10),
    #               -0.2*ones(1, 8),
    #                zeros(1, 9),
    #                zeros(1, 16),
    #                zeros(1, 10)
    #             ),
    #           hcat( 4  *ones(1, 10),
    #                 7.7*ones(1, 15),
    #                    zeros(1, 49),
    #                    zeros(1, 27)
    #                # -8.7*ones(1, 17),
    #                # -5  *ones(1, 10)
    #                )
    #           )

    ws = [w1, w2]
    return ws
end

function get_merging_trajectory_p1_first_101(cfg::MergingScenarioConfig)
    v_init = 10.
    v_goal = 10.
    lw_m = cfg.lane_width_m

    # x₁ = [rlb_x/2; 10.; pi/2; v_init; rlb_x/1.5; 0.; pi/2; v_init]
    x₁ = [-lw_m/2; 15.; pi/2; v_init; lw_m/2; 0.; pi/2; v_init]

    p1_goal = vcat([0.; 95; pi/2; v_goal], zeros(4))
    p2_goal = vcat(zeros(4),               [0.; 80.; pi/2; v_goal])

    # w1 = zeros(2, 101) # Agent 1 keeps going in straight line
    w1 = vcat(hcat(zeros(1, 30),
                  -0.2*ones(1, 8),
                   zeros(1, 10),
                   0.2*ones(1, 8),
                   zeros(1, 9),
                   zeros(1, 16),
                   zeros(1, 20)
                ),
              hcat( 4  *ones(1, 10),
                    7.7*ones(1, 15),
                       zeros(1, 49),
                       zeros(1, 27)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

    # Agent 2 does a passing maneuver.
    w2 = vcat(hcat(zeros(1, 40),
                   0.2*ones(1, 8),
                   zeros(1, 10),
                  -0.2*ones(1, 8),
                   zeros(1, 9),
                   zeros(1, 16),
                   zeros(1, 10)
                ),
              hcat( 4  *ones(1, 10),
                    7.7*ones(1, 15),
                    -7.7*ones(1, 15),
                    7.7*ones(1, 15),
                        zeros(1, 46)
                       # zeros(1, 49),
                       # zeros(1, 27)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

    # vcat(hcat(
    #                zeros(1, 50),
    #                0.4*ones(1, 8),
    #               -0.4*ones(1, 8),
    #                zeros(1, 9),
    #                zeros(1, 16),
    #                zeros(1, 10)), 
    #           hcat( 5  *ones(1, 10),
    #                 8.7*ones(1, 15),
    #                    zeros(1, 49),
    #                    zeros(1, 27)
    #                # -8.7*ones(1, 17),
    #                # -5  *ones(1, 10)
    #                )
    #           )

    ws = [w1, w2]
    return ws, x₁, p1_goal, p2_goal
end

# This one causes p2 to go first.
function get_merging_trajectory_p2_reverse_101(cfg::MergingScenarioConfig)
    v_init = 10.
    lw_m = cfg.lane_width_m

    # x₁ = [rlb_x/2; 10.; pi/2; v_init; rlb_x/1.5; 0.; pi/2; v_init]
    x₁ = [-lw_m/2; 0.; pi/2; v_init; lw_m/2; 15.; pi/2; v_init]

    p1_goal = vcat([0.; 80; pi/2; v_goal], zeros(4))
    p2_goal = vcat(zeros(4),               [0.; 95.; pi/2; v_goal])

    # w1 = zeros(2, 101) # Agent 1 keeps going in straight line
    w1 = vcat(hcat(zeros(1, 40),
                  -0.2*ones(1, 8),
                   zeros(1, 10),
                   0.2*ones(1, 8),
                   zeros(1, 9),
                   zeros(1, 16),
                   zeros(1, 10)
                ),
              hcat( 4  *ones(1, 10),
                    7.7*ones(1, 15),
                    -7.7*ones(1, 15),
                    7.7*ones(1, 15),
                        zeros(1, 46)
                       # zeros(1, 49),
                       # zeros(1, 27)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

    # Agent 2 does a passing maneuver.
    w2 = vcat(hcat(zeros(1, 30),
                   0.2*ones(1, 8),
                   zeros(1, 10),
                  -0.2*ones(1, 8),
                   zeros(1, 9),
                   zeros(1, 16),
                   zeros(1, 20)
                ),
              hcat( 4  *ones(1, 10),
                    7.7*ones(1, 15),
                       zeros(1, 49),
                       zeros(1, 27)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

    # vcat(hcat(
    #                zeros(1, 50),
    #                0.4*ones(1, 8),
    #               -0.4*ones(1, 8),
    #                zeros(1, 9),
    #                zeros(1, 16),
    #                zeros(1, 10)), 
    #           hcat( 5  *ones(1, 10),
    #                 8.7*ones(1, 15),
    #                    zeros(1, 49),
    #                    zeros(1, 27)
    #                # -8.7*ones(1, 17),
    #                # -5  *ones(1, 10)
    #                )
    #           )

    ws = [w1, w2]
    return ws, x₁, p1_goal, p2_goal
end

# Start at the same y-position
function get_merging_trajectory_p1_same_start_101(cfg::MergingScenarioConfig)
    v_init = 10.
    lw_m = cfg.lane_width_m

    x₁ = [-lw_m/2; 0.; pi/2; v_init; lw_m/2; 0.; pi/2; v_init]

    # w1 = zeros(2, 101) # Agent 1 keeps going in straight line
    w1 = vcat(hcat(zeros(1, 30),
                  -0.2*ones(1, 8),
                   zeros(1, 10),
                   0.2*ones(1, 8),
                   zeros(1, 9),
                   zeros(1, 16),
                   zeros(1, 20)
                ),
              hcat( 4  *ones(1, 10),
                    7.7*ones(1, 15),
                       zeros(1, 49),
                       zeros(1, 27)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

    # Agent 2 does a passing maneuver.
    w2 = vcat(hcat(zeros(1, 40),
                   0.2*ones(1, 8),
                   zeros(1, 10),
                  -0.2*ones(1, 8),
                   zeros(1, 9),
                   zeros(1, 16),
                   zeros(1, 10)
                ),
              hcat( 4  *ones(1, 10),
                    7.7*ones(1, 15),
                       zeros(1, 49),
                       zeros(1, 27)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

    ws = [w1, w2]
    return ws, x₁
end

# p1 on right and p2 on left
function get_merging_trajectory_p2_flipped_101(cfg::MergingScenarioConfig)
    v_init = 10.
    lw_m = cfg.lane_width_m

    # x₁ = [rlb_x/2; 10.; pi/2; v_init; rlb_x/1.5; 0.; pi/2; v_init]
    x₁ = [lw_m/2; 0.; pi/2; v_init; -lw_m/2; 15.; pi/2; v_init]

    # w1 = zeros(2, 101) # Agent 1 keeps going in straight line
    w1 = vcat(hcat(zeros(1, 40),
                  -0.2*ones(1, 8),
                   zeros(1, 10),
                   0.2*ones(1, 8),
                   zeros(1, 9),
                   zeros(1, 16),
                   zeros(1, 10)
                ),
              hcat( 4  *ones(1, 10),
                    7.7*ones(1, 15),
                       zeros(1, 49),
                       zeros(1, 27)
                    # -7.7*ones(1, 15),
                    # 7.7*ones(1, 15),
                    #     zeros(1, 46)
                       # zeros(1, 49),
                       # zeros(1, 27)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

    # Agent 2 does a passing maneuver.
    w2 = vcat(hcat(zeros(1, 30),
                   0.2*ones(1, 8),
                   zeros(1, 10),
                  -0.2*ones(1, 8),
                   zeros(1, 9),
                   zeros(1, 16),
                   zeros(1, 20)
                ),
              hcat( 4  *ones(1, 10),
                    7.7*ones(1, 15),
                    -7.7*ones(1, 15),
                    7.7*ones(1, 15),
                        zeros(1, 46)
                       # zeros(1, 49),
                       # zeros(1, 27)
                   # -8.7*ones(1, 17),
                   # -5  *ones(1, 10)
                   )
              )

    # vcat(hcat(
    #                zeros(1, 50),
    #                0.4*ones(1, 8),
    #               -0.4*ones(1, 8),
    #                zeros(1, 9),
    #                zeros(1, 16),
    #                zeros(1, 10)), 
    #           hcat( 5  *ones(1, 10),
    #                 8.7*ones(1, 15),
    #                    zeros(1, 49),
    #                    zeros(1, 27)
    #                # -8.7*ones(1, 17),
    #                # -5  *ones(1, 10)
    #                )
    #           )

    ws = [w2, w1]
    return ws, x₁, p1_goal, p2_goal
end



# For saving the trajectory - in case we want to use it later.
function save_trajectory(filename, xs, us, times, T, dt)
    # Create some data to save
    data = Dict("xs" => xs, "us" => us, "times" => times, "T" => T, "dt" => dt)

    # Save the data to a file
    @save "$(filename).jld" data
end

function load_trajectory(filename)
    @load "$(filename).jld" data

    return data["xs"], data["us"], data["times"], data["T"], data["dt"]
end


function generate_gt_from_silqgames(sg_obj, leader_idx::Int, times, x₁, us_1)

    # ground truth generation config variables
    leader_idx = 1

    t₀ = times[1]

    xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, t₀, times, x₁, us_1)

    println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
    final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
    println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals, " sum: ", sum(final_cost_totals))

    return xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs
end

function plot_silqgames_gt(dyn, cfg::MergingScenarioConfig, times, xs_k, us_k)
    # PLOTS A: Plot states/controls.
    l = @layout [
        a{0.3h}; [grid(2,3)]
    ]

    # q = @layout [a b; c d ;e f; g h]
    pos_plot, p2, p3, p4, p5, p6, p7 = plot_states_and_controls(dyn, times, xs_k, us_k)
    # plot(pos_plot, p2, p3, p4, p5, p6, p7, layout = l)
    plt = plot(pos_plot, size=(300, 800))

    w = cfg.lane_width_m
    L₁ = cfg.region1_length_m
    L₂ = cfg.region2_length_m
    plot!(pos_plot, [w, w], [0., L₁], label="", color=:black, lw=3)
    plot!(pos_plot, [0, 0], [0., L₁], label="", color=:black, lw=3)
    plot!(pos_plot, [-w, -w], [0., L₁], label="", color=:black, lw=3)
    plot!(pos_plot, [w, w/2], [L₁, L₁+L₂], label="", color=:black, lw=3)
    plot!(pos_plot, [-w, -w/2], [L₁, L₁+L₂], label="", color=:black, lw=3)
    plot!(pos_plot, [w/2, w/2], [L₁+L₂, 2*(L₁+L₂)], label="", color=:black, lw=3)
    plot!(pos_plot, [-w/2, -w/2], [L₁+L₂, 2*(L₁+L₂)], label="", color=:black, lw=3)

    plot!(pos_plot, [])
end

function plot_silqgames_gt_convergence_cost(is_converged, num_iters, conv_metrics, evaluated_costs)
    # PLOTS B: Plot convergence metrics/costs separately.
    conv_plot, cost_plot = plot_convergence_and_costs(num_iters, threshold, conv_metrics, evaluated_costs)
    m = @layout [a; b]
    plot(conv_plot, cost_plot, layout = m)
end
