# Unit tests for LQ Nash solvers.
using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Random: seed!
using Test: @test, @testset
include("TestUtils.jl")

seed!(0)

# These tests run checks to ensure that for the LQ case given appropriate reference trajectories, the approximate
# linearized, quadraticized solutions are identical to those from the LQ solvers.
@testset "TestApproximatedLQSolvers" begin
    stackelberg_leader_idx = 1

    t0 = 0.0
    x₁ = [1.; 0; 1.; 0]
    x₁ = vcat(x₁, x₁)
    num_states = size(x₁, 1)
    horizon = 10
    times = cumsum(ones(horizon)) .- 1 .+ t0

    # Ensure that for an LQ optimal control problem, both the LQ optimal control solution and approximate LQ Nash
    # optimal control with the solution to the LQ optimal control problem as reference, are identical!
    @testset "CheckLQOptCtrlAndApproximatedLQOptCtrlGiveSameOutputForLQGame" begin
        sys_info = SystemInfo(1, num_states, [2])
        dyn = generate_random_linear_dynamics(sys_info)
        costs = generate_random_quadratic_costs(sys_info; include_cross_costs=true, make_affine=true)

        Ps, future_costs_expected = solve_lqr_feedback(dyn, costs[1], horizon)
        xs, us = unroll_feedback(dyn, times, FeedbackGainControlStrategy([Ps]), x₁)
        P̃s, future_costs_actual = solve_approximated_lqr_feedback(dyn, costs[1], horizon, t0, xs, us[1])

        @test Ps == P̃s
        @test all([future_costs_expected[tt].Q == future_costs_actual[tt].Q for tt in 1:horizon])
    end

    # Ensure that for an LQ game, both the LQ Nash solution and approximate LQ Nash solution with the solution to the
    # LQ Nash game as reference, are identical!
    @testset "CheckLQNashAndApproximatedLQNashGiveSameOutputForLQGame" begin
        num_players = 2
        num_ctrls = [2, 3]
        sys_info = SystemInfo(num_players, num_states, num_ctrls)

        # Generate the game.
        dyn = generate_random_linear_dynamics(sys_info)
        costs = generate_random_quadratic_costs(sys_info; include_cross_costs=true, make_affine=true)

        Ps, future_costs_expected = solve_lq_nash_feedback(dyn, costs, horizon)
        xs, us = unroll_feedback(dyn, times, FeedbackGainControlStrategy(Ps), x₁)
        P̃s, future_costs_actual = solve_approximated_lq_nash_feedback(dyn, costs, horizon, t0, xs, us)

        @test Ps == P̃s
        for ii in 1:num_players
            @test all([future_costs_expected[ii][tt].Q == future_costs_actual[ii][tt].Q for tt in 1:horizon])
        end
    end

    # Ensure that for an LQ game, both the LQ Stackelberg solution and approximate LQ Stackelberg solution with the solution to the
    # LQ Stackelberg game as reference, are identical!
    @testset "CheckLQStackelbergAndApproximatedLQStackelbergGiveSameOutputForLQGame" begin
        stackelberg_leader_idx = 1

        num_players = 2
        num_ctrls = [2, 3]
        sys_info = SystemInfo(num_players, num_states, num_ctrls)

        # Generate the game.
        dyn = generate_random_linear_dynamics(sys_info)
        costs = generate_random_quadratic_costs(sys_info; include_cross_costs=true, make_affine=true)

        Ss, future_costs_expected = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
        xs, us = unroll_feedback(dyn, times, FeedbackGainControlStrategy(Ss), x₁)
        S̃s, future_costs_actual = solve_approximated_lq_stackelberg_feedback(dyn, costs, horizon, t0, xs, us, stackelberg_leader_idx)

        @test Ss == S̃s
        for ii in 1:num_players
            @test all([future_costs_expected[ii][tt].Q == future_costs_actual[ii][tt].Q for tt in 1:horizon])
        end
    end
end
