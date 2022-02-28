# Unit tests for LQ Nash solvers.

using StackelbergControlHypothesesFiltering
using Test: @test, @testset
using Random: seed!

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
    dyn = Dynamics(A, [B₁, B₂])

    Q₁ = [0 0 0  0;
          0 0 0  0;
          0 0 1. 0;
          0 0 0  0]
    c₁ = Cost(Q₁)
    add_control_cost!(c₁, 1, ones(1, 1))

    Q₂ = [1.  0 -1 0;
          0  0 0  0;
          -1 0 1  0;
          0  0 0  0]
    c₂ = Cost(Q₂)
    add_control_cost!(c₂, 2, ones(1, 1))

    costs = [c₁, c₂]

    x₁ = [1., 0, 1, 0]
    horizon = 10


    # Ensure that the feedback solution satisfies Nash conditions of optimality
    # for each player, holding others' strategies fixed.
    @testset "CheckFeedbackSatisfiesNash" begin
        Ps = solve_lq_nash_feedback(dyn, costs, horizon)
        xs, us = unroll_feedback(dyn, Ps, x₁)
        nash_costs = [evaluate(c, xs, us) for c in costs]

        # Perturb each strategy a little bit and confirm that cost only
        # increases for that player.
        ϵ = 1e-1
        for ii in 1:2
            for tt in 1:horizon
                P̃s = deepcopy(Ps)
                P̃s[ii][:, :, tt] += ϵ * randn(udim(dyn, ii), xdim(dyn))

                x̃s, ũs = unroll_feedback(dyn, P̃s, x₁)
                new_nash_costs = [evaluate(c, x̃s, ũs) for c in costs]
                @test new_nash_costs[ii] ≥ nash_costs[ii]
            end
        end
    end


    # Ensure that the feedback solution satisfies Stackelberg conditions of optimality
    # for each player, holding others' strategies fixed.
    @testset "CheckFeedbackSatisfiesStackelberg" begin
        Ss = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
        xs, us = unroll_feedback(dyn, Ss, x₁)
        optimal_stackelberg_costs = [evaluate(c, xs, us) for c in costs]

        # Perturb each P1 strategy a little bit, resolve for P2's input, and ensure the optimal cost for P1 is less than
        # the alterred cost for P1.
        ϵ = 1e-1
        leader_idx = 1
        follower_idx = 2
        num_players = follower_idx

        # Feedback at horizon is state cost.
        P2ₜ₊₁ = costs[follower_idx].Q

        # At each time step running backwards, perturb P1 at tt, solve the LQR problem for P2 with it, and validate the
        # new costs are higher than the old ones.
        for tt in horizon-1:-1:1

            # Optimal P2s given optimal P1s.
            P̃2s = deepcopy(Ss)[follower_idx]

            # Reset P1s to the optimal P1s and add noise to P1's feedback matrices at tt.
            P̃1s = deepcopy(Ss)[leader_idx]
            P̃1s[:, :, tt] += ϵ * randn(udim(dyn, leader_idx), xdim(dyn))

            # Create new dynamics and cost objects for use in the LQR solver.
            F = dyn.A - dyn.Bs[leader_idx] * P̃1s[:, :, tt]
            dyn_P2_lqr = Dynamics(F, [dyn.Bs[follower_idx]])
            costs_P2_lqr = Cost(costs[follower_idx].Q)
            add_control_cost!(costs_P2_lqr, 1, costs[follower_idx].Rs[follower_idx])

            # Solve the LQR problem from tt to horizon. Since P2 and P1 are optimal until tt, these should be otherwise
            # the same before tt.
            P̃2s[:, :, tt:horizon] = solve_lqr_feedback(dyn_P2_lqr, costs_P2_lqr, horizon - tt + 1)

            # Unroll the states again from the initial state.
            P̃s = [P̃1s, P̃2s]
            x̃s, ũs = unroll_feedback(dyn, P̃s, x₁)

            # Validate the new costs are higher than the optimal ones for the trajectory up to tt.
            new_stack_costs = [evaluate(c, x̃s, ũs) for c in costs]
            optimal_stackelberg_costs = [evaluate(c, xs, us) for c in costs]
            @test new_stack_costs[leader_idx] ≥ optimal_stackelberg_costs[leader_idx]
        end

        # Perturb each optimized P2 strategy a little bit and confirm that cost only increases for player 2.
        ii = 2
        jj = 1
        for tt in 1:horizon
            P̃s = deepcopy(Ss)
            P̃s[ii][:, :, tt] += ϵ * randn(udim(dyn, ii), xdim(dyn))

            x̃s, ũs = unroll_feedback(dyn, P̃s, x₁)
            new_stack_costs = [evaluate(c, x̃s, ũs) for c in costs]
            @test new_stack_costs[ii] ≥ optimal_stackelberg_costs[ii]
        end

    end

end
