# Unit tests for LQ Nash solvers.
using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Random: seed!
using Test: @test, @testset

seed!(0)

@testset "TestLQSolvers" begin
    stackelberg_leader_idx = 1

    # Common dynamics, costs, and initial condition.
    A = [1 0.1 0 0;
         0 1   0 0;
         0 0   1 1.1;
         0 0   0 1]
    B₁ = [0 0.1 0 0]'
    B₂ = [0 0   0 0.1]'
    dyn = LinearDynamics(A, [B₁, B₂])

    Q₁ = [0 0 0  0;
          0 0 0  0;
          0 0 1. 0;
          0 0 0  0]
    c₁ = QuadraticCost(Q₁)
    add_control_cost!(c₁, 1, ones(1, 1))
    add_control_cost!(c₁, 2, zeros(1, 1))

    Q₂ = [1.  0 -1 0;
          0  0 0  0;
          -1 0 1  0;
          0  0 0  0]
    c₂ = QuadraticCost(Q₂)
    add_control_cost!(c₂, 2, ones(1, 1))
    add_control_cost!(c₂, 1, zeros(1, 1))

    x₁ = [1., 0, 1, 0]
    horizon = 100
    times = cumsum(ones(horizon)) .- 1.

    costs = [c₁, c₂]

     @testset "CheckFeedbackSatisfiesLQROptimum" begin
        # This is similar to the LQR problem that MATLAB has as an example - dt = 0.02s and it's a double integrator
        # with state [x ẋ] and input control α acceleration. We've added a constrant drift of +1 m and +1 m/s in
        # dynamics.
        dt = 0.02
        times_lqr = dt * (cumsum(ones(horizon)) .- 1.)
        lin_dyn = LinearDynamics([1. dt; 0 1.], [[0.; dt][:,:]]; a=ones(2))
        q_cost = QuadraticCost([1. 0.; 0. 0.], ones(2), 1.)
        add_control_cost!(q_cost, 1, ones(1, 1); r=zeros(1), cr=2.)


        # lin_dyn = LinearDynamics([1. dt; 0 1.], [[0.; dt][:,:]])
        # q_cost = QuadraticCost([1. 0.; 0. 0.])
        # add_control_cost!(q_cost, 1, ones(1, 1))
        new_x₁ = ones(2)

        strategy, _ = solve_lqr_feedback(lin_dyn, q_cost, horizon)
        xs, us = unroll_feedback(lin_dyn, times_lqr, strategy, new_x₁)
        eval_cost = evaluate(q_cost, xs, us)

        # Perturb each strategy a little bit and confirm that cost only
        # increases for that player.
        ϵ = 1e-1
        for tt in 1:horizon
            P̃s = deepcopy(get_linear_feedback_gain(strategy, 1))
            p̃s = deepcopy(get_constant_feedback_gain(strategy, 1))
            P̃s[:, :, tt] += ϵ * randn(udim(lin_dyn, 1), xdim(lin_dyn))
            p̃s[:, tt] += ϵ * randn(udim(lin_dyn, 1))

            x̃s, ũs = unroll_feedback(lin_dyn, times, FeedbackGainControlStrategy([P̃s], [p̃s]), new_x₁)
            new_eval_cost = evaluate(q_cost, x̃s, ũs)
            @test new_eval_cost ≥ eval_cost
        end
    end

    # Ensure that the costs match up at each time step with manually calculate cost matrices.
    @testset "CheckLQRCostsAreConsistentAtEquilibrium" begin
        # This is the same LQR problem that MATLAB has as an example - dt = 0.02s and it's a double integrator with
        # state [x ẋ] and input control α acceleration.
        dt = 0.02
        times_lqr = dt * (cumsum(ones(horizon)) .- 1.)
        lin_dyn = LinearDynamics([1. dt; 0 1.], [[0.; dt][:,:]]; a=zeros(2))
        q_cost = QuadraticCost([1. 0.; 0. 0.], zeros(2), 1.)
        add_control_cost!(q_cost, 1, ones(1, 1); r=zeros(1), cr=2.)
        new_x₁ = ones(2)

        strategy, future_costs = solve_lqr_feedback(lin_dyn, q_cost, horizon)
        xs, us = unroll_feedback(lin_dyn, times_lqr, strategy, new_x₁)
        eval_cost = evaluate(q_cost, xs, us)

        # Compute the costs using the t+1 cost matrix and compare with the cost using the cost matrix at time t.
        num_players = num_agents(lin_dyn)

        # Homgenize states and controls.
        # xhs = homogenize_vector(xs)

        for tt in 1:horizon-1
            time_range = (tt, tt+1)

            u_tt = us[1][:, tt]
            u_ttp1 = us[1][:, tt+1]

            # TODO(hamzah) Fix discrepancy in extra cost in quad cost.

            # Manual cost is formed by the sum of the current state/ctrls costs and the future costs.
            manual_cost = compute_cost(q_cost, time_range, xs[:, tt], [u_tt])
            manual_cost += compute_cost(future_costs[tt+1], time_range, xs[:, tt+1], [u_ttp1])
            computed_cost = compute_cost(future_costs[tt], time_range, xs[:, tt], [u_tt])

            @test manual_cost ≈ computed_cost
        end
    end

    # # Ensure that the feedback solution satisfies Nash conditions of optimality
    # # for each player, holding others' strategies fixed.
    # # Note: This test, as formulated, allows some false positive cases. See Basar and Olsder (Eq. 3.22) for the exact
    # #       conditions.
    # @testset "CheckFeedbackSatisfiesNash" begin
    #     ctrl_strat_Ps, _ = solve_lq_nash_feedback(dyn, costs, horizon)
    #     xs, us = unroll_feedback(dyn, times, ctrl_strat_Ps, x₁)
    #     nash_costs = [evaluate(c, xs, us) for c in costs]

    #     # Perturb each strategy a little bit and confirm that cost only
    #     # increases for that player.
    #     ϵ = 1e-1
    #     for ii in 1:2
    #         for tt in 1:horizon
    #             ctrl_strat_P̃s = deepcopy(ctrl_strat_Ps)
    #             ctrl_strat_P̃s.Ps[ii][:, :, tt] += ϵ * randn(udim(dyn, ii), xdim(dyn))
    #             ctrl_strat_P̃s.ps[ii][:, tt] += ϵ * randn(udim(dyn, ii))

    #             x̃s, ũs = unroll_feedback(dyn, times, ctrl_strat_P̃s, x₁)
    #             new_nash_costs = [evaluate(c, x̃s, ũs) for c in costs]
    #             @test new_nash_costs[ii] ≥ nash_costs[ii]
    #         end
    #     end
    # end


    # # Ensure that the costs match up at each time step with manually calculate cost matrices.
    # @testset "CheckNashCostsAreConsistentAtEquilibrium" begin
    #     ctrl_strat_Ps, future_costs = solve_lq_nash_feedback(dyn, costs, horizon)
    #     xs, us = unroll_feedback(dyn, times, ctrl_strat_Ps, x₁)

    #     # Compute the costs using the t+1 cost matrix and compare with the cost using the cost matrix at time t.
    #     num_players = num_agents(dyn)

    #     for ii in 1:2
    #         for tt in 1:horizon-1
    #             time_range = (tt, tt+1)

    #             u_tt = [us[ii][:, tt] for ii in 1:num_players]
    #             u_ttp1 = [us[ii][:, tt+1] for ii in 1:num_players]

    #             # TODO(hamzah) Fix discrepancy in extra cost in quad cost.

    #             # Manual cost is formed by the sum of the current state/ctrls costs and the future costs.
    #             manual_cost = compute_cost(costs[ii], time_range, xs[:, tt], u_tt)
    #             manual_cost += compute_cost(future_costs[ii][tt+1], time_range, xs[:, tt+1], u_ttp1)
    #             computed_cost = compute_cost(future_costs[ii][tt], time_range, xs[:, tt], u_tt)

    #             @test manual_cost ≈ computed_cost
    #         end
    #     end
    # end


    # # Ensure that the feedback solution satisfies Stackelberg conditions of optimality
    # # for player 1, holding others' strategies fixed.
    # # Note: This test, as formulated, allows some false positive cases. See Khan and Fridovich-Keil 2023 for the exact
    # #       conditions.
    # @testset "CheckFeedbackSatisfiesStackelbergEquilibriumForLeader" begin
    #     ctrl_strat_Ss, future_costs = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
    #     xs, us = unroll_feedback(dyn, times, ctrl_strat_Ss, x₁)
    #     optimal_stackelberg_costs = [evaluate(c, xs, us) for c in costs]

    #     # Define some useful constants.
    #     ϵ = 1e-1
    #     leader_idx = stackelberg_leader_idx
    #     follower_idx = 3 - stackelberg_leader_idx
    #     num_players = num_agents(dyn)

    #     for tt in horizon-1:-1:1
    #         time_range = (tt, tt+1)

    #         # Copy the things we will alter.
    #         ũs = deepcopy(us)

    #         # In this test, we need to solve the "optimal control" problem for the second player given player 1's
    #         # control.

    #         # Perturb the leader input u1 at the current time.
    #         ũs[leader_idx][:, tt] += ϵ * randn(udim(dyn, leader_idx))
    #         ũhs = homogenize_ctrls(dyn, ũs)

    #         # Re-solve for the optimal follower input given the perturbed leader trajectory.
    #         A = get_homogenized_state_dynamics_matrix(dyn)
    #         B₂ = get_homogenized_control_dynamics_matrix(dyn, follower_idx)
    #         L₂_ttp1 = get_homogenized_state_cost_matrix(future_costs[follower_idx][tt+1])
    #         G = get_homogenized_control_cost_matrix(costs[follower_idx], follower_idx) + B₂' * L₂_ttp1 * B₂

    #         # Homogenize states and controls.
    #         xhs = homogenize_vector(xs)

    #         B₁ = get_homogenized_control_dynamics_matrix(dyn, leader_idx)
    #         ũh1ₜ = ũhs[leader_idx][:, tt]
    #         ũh2ₜ = - G \ (B₂' * L₂_ttp1 * (A * xhs[:,tt] + B₁ * ũh1ₜ))
    #         ũ_tt = [ũh1ₜ[1:udim(dyn, 1)], ũh2ₜ[1:udim(dyn, 2)]]

    #         ũhs[follower_idx][:, tt] = ũh2ₜ

    #         # The cost of the solution trajectory, computed as x_t^T * L^1_tt x_t for at time tt.
    #         # We test the accuracy of this cost in `CheckStackelbergCostsAreConsistentAtEquilibrium`.
    #         opt_P1_cost = compute_cost(future_costs[leader_idx][tt], time_range, xs[:, tt], ũ_tt)

    #         # Compute the homogenized controls for time tt+1.
    #         u_ttp1 = [ũhs[ii][:, tt+1][1:udim(dyn, ii)] for ii in 1:num_players]

    #         # The cost computed manually for perturbed inputs using
    #         # x_t^T Q_t x_t^T + ... + <control costs> + ... + x_{t+1}^T * L^1_{t+1} x_{t+1}.
    #         state_and_controls_cost = compute_cost(costs[leader_idx], time_range, xs[:, tt], ũ_tt)
    #         ũ_tt = [ũs[ii][:, tt] for ii in 1:num_players]
    #         xₜ₊₁ = propagate_dynamics(dyn, time_range, xs[:, tt], ũ_tt)
    #         future_cost = compute_cost(future_costs[leader_idx][tt+1], time_range, xₜ₊₁, u_ttp1)
    #         new_P1_cost = state_and_controls_cost + future_cost

    #         # The costs from time t+1 of the perturbed and optimal trajectories should also satisfy this condition.
    #         @test new_P1_cost ≥ opt_P1_cost

    #         x̃s = unroll_raw_controls(dyn, times, ũs, x₁)
    #         new_stack_costs = [evaluate(c, x̃s, ũs) for c in costs]
    #         optimal_stackelberg_costs = [evaluate(c, xs, us) for c in costs]

    #         # This test evaluates the cost for the entire perturbed trajectory against the optimal cost.
    #         @test new_stack_costs[leader_idx] ≥ optimal_stackelberg_costs[leader_idx]
    #     end
    # end


    # # Ensure that the feedback solution satisfies Stackelberg conditions of optimality
    # # for player 2, holding others' strategies fixed.
    # @testset "CheckFeedbackSatisfiesStackelbergEquilibriumForFollower" begin
    #     ctrl_strat_Ss, future_costs = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
    #     xs, us = unroll_feedback(dyn, times, ctrl_strat_Ss, x₁)
    #     optimal_stackelberg_costs = [evaluate(c, xs, us) for c in costs]

    #     # Define some useful constants.
    #     ϵ = 1e-1
    #     leader_idx = stackelberg_leader_idx
    #     follower_idx = 3 - stackelberg_leader_idx
    #     num_players = follower_idx

    #     # Perturb each optimized P2 strategy a little bit and confirm that cost only increases for player 2.
    #     leader_idx = 1
    #     follower_idx = 2
    #     for tt in 1:horizon-1
    #         ctrl_strat_S̃s = deepcopy(ctrl_strat_Ss)
    #         ctrl_strat_S̃s.Ps[follower_idx][:, :, tt] += ϵ * randn(udim(dyn, follower_idx), xdim(dyn))
    #         ctrl_strat_S̃s.ps[follower_idx][:, tt] += ϵ * randn(udim(dyn, follower_idx))

    #         x̃s, ũs = unroll_feedback(dyn, times, ctrl_strat_S̃s, x₁)
    #         new_stack_costs = [evaluate(c, x̃s, ũs) for c in costs]
    #         @test new_stack_costs[follower_idx] ≥ optimal_stackelberg_costs[follower_idx]
    #     end
    # end


    # # Ensure that the costs match up at each time step with manually calculate cost matrices.
    # @testset "CheckStackelbergCostsAreConsistentAtEquilibrium" begin
    #     ctrl_strat_Ss, future_costs = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
    #     xs, us = unroll_feedback(dyn, times, ctrl_strat_Ss, x₁)

    #     # For each player, compute the costs using the t+1 cost matrix and compare with the cost using the cost matrix
    #     # at time t.

    #     # Compute the costs using the t+1 cost matrix and compare with the cost using the cost matrix at time t.
    #     num_players = num_agents(dyn)

    #     for ii in 1:2
    #         jj = 3 - ii
    #         for tt in 1:horizon-1
    #             time_range = (tt, tt+1)

    #             u_tt = [us[ii][:, tt] for ii in 1:num_players]
    #             # uh_tt = homogenize_ctrls(dyn, u_tt)

    #             u_ttp1 = [us[ii][:, tt+1] for ii in 1:num_players]
    #             # uh_ttp1 = homogenize_ctrls(dyn, u_ttp1)

    #             # TODO(hamzah) Fix discrepancy in extra cost in quad cost.
    #             state_and_control_costs = compute_cost(costs[ii], time_range, xs[:, tt], u_tt)
    #             future_cost = compute_cost(future_costs[ii][tt+1], time_range, xs[:, tt+1], u_ttp1)

    #             manual_cost = state_and_control_costs + future_cost
    #             computed_cost = compute_cost(future_costs[ii][tt], time_range, xs[:, tt], u_tt)

    #             # The manually recursion at time t should match the computed L cost at time t.
    #             @test manual_cost ≈ computed_cost
    #         end
    #     end
    # end
end
