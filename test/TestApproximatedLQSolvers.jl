# Unit tests for approximated solvers that INCORRECTLY linearize/quadraticize - see iLQR, ILQGames, Stackelberg ILQGames.
using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Random: seed!
using Test: @test, @testset
include("TestUtils.jl")

seed!(0)


# Solve a finite horizon, discrete time LQR problem by approximating non-LQ dynamics/costs as LQ at each timestep.
# Returns feedback matrices P[:, :, time].

# A function which accepts non-linear dynamics and non-quadratic costs and solves an LQ approximation at each timestep.
function solve_approximated_lqr_feedback(dyn::Dynamics,
                                         cost::Cost,
                                         horizon::Int,
                                         t0::Float64,
                                         xs_1::AbstractArray{Float64},
                                         us_1::AbstractArray{Float64})
    T = horizon
    N = num_agents(dyn)

    lin_dyns = Array{LinearDynamics}(undef, T)
    quad_costs = Array{QuadraticCost}(undef, T)

    for tt in 1:T
        prev_time = t0 + ((tt == 1) ? 0 : tt-1)
        current_time = t0 + tt
        time_range = (prev_time, current_time)
        lin_dyns[tt] = linearize_dynamics(dyn, time_range, xs_1[:, tt], [us_1[:, tt]])
        quad_costs[tt] = quadraticize_costs(cost, time_range, xs_1[:, tt], [us_1[:, tt]])
    end

    return solve_lqr_feedback(lin_dyns, quad_costs, T)
end


# A function which accepts non-linear dynamics and non-quadratic costs and solves an LQ approximation at each timestep.
function solve_approximated_lq_nash_feedback(dyn::Dynamics,
                                             costs::AbstractVector{<:Cost},
                                             horizon::Int,
                                             t0::Float64,
                                             x_refs::AbstractArray{Float64},
                                             u_refs::AbstractVector{<:AbstractArray{Float64}})
    T = horizon
    N = num_agents(dyn)

    lin_dyns = Vector{LinearDynamics}(undef, T)
    all_quad_costs = Vector{Vector{QuadraticCost}}(undef, T)

    for tt in 1:T
        prev_time = t0 + ((tt == 1) ? 0 : tt-1)

        quad_costs = Vector{QuadraticCost}(undef, N)
        u_refs_at_tt = [u_refs[ii][:, tt] for ii in 1:N]
        current_time = t0 + tt

        # Linearize and quadraticize the dynamics/costs.
        time_range = (prev_time, current_time)
        lin_dyns[tt] = linearize_dynamics(dyn, time_range, x_refs[:, tt], u_refs_at_tt)
        for ii in 1:N
            quad_costs[ii] = quadraticize_costs(costs[ii], time_range, x_refs[:, tt], u_refs_at_tt)
        end
        all_quad_costs[tt] = quad_costs
    end

    return solve_lq_nash_feedback(lin_dyns, all_quad_costs, T)
end


# A function which accepts non-linear dynamics and non-quadratic costs and solves an LQ approximation at each timestep.
function solve_approximated_lq_stackelberg_feedback(dyn::Dynamics,
                                                    costs::AbstractVector{<:Cost},
                                                    horizon::Int,
                                                    t0::Float64,
                                                    x_refs::AbstractArray{Float64},
                                                    u_refs::AbstractVector{<:AbstractArray{Float64}},
                                                    leader_idx::Int)
    T = horizon
    N = num_agents(dyn)

    lin_dyns = Vector{LinearDynamics}(undef, T)
    all_quad_costs = Vector{Vector{QuadraticCost}}(undef, T)

    for tt in 1:T
        prev_time = t0 + ((tt == 1) ? 0 : tt-1)
        current_time = t0 + tt
        time_range = (prev_time, current_time)

        quad_costs = Vector{QuadraticCost}(undef, N)
        u_refs_at_tt = [u_refs[ii][:, tt] for ii in 1:N]

        # Linearize and quadraticize the dynamics/costs.
        lin_dyns[tt] = linearize_dynamics(dyn, time_range, x_refs[:, tt], u_refs_at_tt)
        for ii in 1:N
            quad_costs[ii] = quadraticize_costs(costs[ii], time_range, x_refs[:, tt], u_refs_at_tt)
        end
        all_quad_costs[tt] = quad_costs
    end

    return solve_lq_stackelberg_feedback(lin_dyns, all_quad_costs, T, leader_idx)
end


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
        costs = generate_random_quadratic_costs(sys_info; include_cross_costs=true, make_affine=false)

        ctrl_strat_Ps, future_costs_expected = solve_lqr_feedback(dyn, costs[1], horizon)
        xs, us = unroll_feedback(dyn, times, ctrl_strat_Ps, x₁)
        ctrl_strat_P̃s, future_costs_actual = solve_approximated_lqr_feedback(dyn, costs[1], horizon, t0, xs, us[1])

        println(norm(ctrl_strat_Ps.Ps), " ", norm(ctrl_strat_P̃s.Ps))
        println(norm(ctrl_strat_Ps.ps), " ", norm(ctrl_strat_P̃s.ps))
        @test ctrl_strat_Ps.Ps == ctrl_strat_P̃s.Ps && ctrl_strat_Ps.ps == ctrl_strat_P̃s.ps
        # Compare the Q matrices.
        @test all([get_homogenized_state_cost_matrix(future_costs_expected[tt]) == get_homogenized_state_cost_matrix(future_costs_actual[tt]) for tt in 1:horizon])
    end

    # Ensure that for an LQ game, both the LQ Nash solution and approximate LQ Nash solution with the solution to the
    # LQ Nash game as reference, are identical!
    @testset "CheckLQNashAndApproximatedLQNashGiveSameOutputForLQGame" begin
        num_players = 2
        num_ctrls = [2, 3]
        sys_info = SystemInfo(num_players, num_states, num_ctrls)

        # Generate the game.
        dyn = generate_random_linear_dynamics(sys_info)
        costs = generate_random_quadratic_costs(sys_info; include_cross_costs=true, make_affine=false)

        ctrl_strats_Ps, future_costs_expected = solve_lq_nash_feedback(dyn, costs, horizon)
        xs, us = unroll_feedback(dyn, times, ctrl_strats_Ps, x₁)
        ctrl_strats_P̃s, future_costs_actual = solve_approximated_lq_nash_feedback(dyn, costs, horizon, t0, xs, us)

        @test ctrl_strats_Ps.Ps == ctrl_strats_P̃s.Ps && ctrl_strats_Ps.ps == ctrl_strats_P̃s.ps
        for ii in 1:num_players
            # Compare the Q matrices.
            @test all([get_homogenized_state_cost_matrix(future_costs_expected[ii][tt]) == get_homogenized_state_cost_matrix(future_costs_actual[ii][tt]) for tt in 1:horizon])
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
        costs = generate_random_quadratic_costs(sys_info; include_cross_costs=true, make_affine=false)

        ctrl_strats_Ss, future_costs_expected = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
        xs, us = unroll_feedback(dyn, times, ctrl_strats_Ss, x₁)
        ctrl_strats_S̃s, future_costs_actual = solve_approximated_lq_stackelberg_feedback(dyn, costs, horizon, t0, xs, us, stackelberg_leader_idx)

        @test ctrl_strats_Ss.Ps == ctrl_strats_S̃s.Ps && ctrl_strats_Ss.ps == ctrl_strats_S̃s.ps
        for ii in 1:num_players
            # Compare the Q matrices.
            @test all([get_homogenized_state_cost_matrix(future_costs_expected[ii][tt]) == get_homogenized_state_cost_matrix(future_costs_actual[ii][tt]) for tt in 1:horizon])
        end
    end
end
