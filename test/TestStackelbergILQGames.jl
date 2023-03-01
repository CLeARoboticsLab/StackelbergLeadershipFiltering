# Unit tests for LQ Nash solvers.
using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Random: seed!
using Test: @test, @testset
include("TestUtils.jl")

seed!(0)

# These tests run checks to ensure that for the LQ case given appropriate reference trajectories, the approximate
# linearized, quadraticized solutions are identical to those from the LQ solvers.
@testset "TestStackelbergILQGames" begin
    stackelberg_leader_idx = 1

    t0 = 0.0
    x₁ = [1.; 0; 1.; 0]
    x₁ = vcat(x₁, x₁)
    num_states = size(x₁, 1)
    horizon = 10

    # Ensure that for an LQ game, both the LQ Stackelberg solution and approximate LQ Stackelberg solution with the solution to the
    # LQ Stackelberg game as reference, are identical!
    @testset "CheckStackILQGamesConvergesInOneIterationForLQGameWithSolutionReference" begin

        num_players = 2
        num_ctrls = [2, 3]
        sys_info = SystemInfo(num_players, num_states, num_ctrls)

        # Generate the game.
        dyn = generate_random_linear_dynamics(sys_info)
        costs = generate_random_quadratic_costs(sys_info; include_cross_costs=true, make_affine=true)

        # Expected solutions.
        Ss, Ls = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
        xs, us = unroll_feedback(dyn, FeedbackGainControlStrategy(Ss), x₁)

        initial_ctrl_strats = StackelbergControlStrategy(horizon, Ss, Ls)
        current_ctrl_strats, is_converged, num_iters = stackelberg_ilqgames(stackelberg_leader_idx, 
                                                                            horizon,
                                                                            x₁,
                                                                            t0,
                                                                            dyn,
                                                                            costs,
                                                                            initial_ctrl_strats,
                                                                            xs,
                                                                            us)

        S̃s, L̃s = current_ctrl_strats.Ss, current_ctrl_strats.Ls

        @test num_iters == 1
        @test Ss == S̃s
        @test Ls == L̃s
    end
end
