# Unit tests for LQ Nash solvers.
using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Random: seed!
using Test: @test, @testset

include("TestUtils.jl")

seed!(0)

function generate_random_state_ctrl_point(si::SystemInfo)
    x0 = rand(xdim(si))
    u0s = [rand(udim(si, ii)) for ii in 1:num_agents(si)]
    return x0, u0s
end

# Provide an initial state/control and generate a control-state trajectory from it.
# Also requires a system info and horizon
function generate_random_infeasible_trajectory(si::SystemInfo, x0, u0s, horizon)
    # Generate states and controls.
    xs = hcat(x0, rand(xdim(si), horizon-1))
    us = [hcat(u0s[ii], rand(udim(si, ii), horizon-1))  for ii in 1:num_agents(si)]
    return xs, us
end

# Tests that the costs have the same results across all timepoints and over the entire trajectory.
function ensure_same_costs(cost1::Cost, cost2::Cost, horizon, time_range, xs, us)
    num_players = size(us, 1)
    for tt in 1:horizon
        x_at_tt = xs[:, tt]
        us_at_tt = [us[ii][:, tt] for ii in 1:num_players]
        @test compute_cost(cost1, time_range, x_at_tt, us_at_tt) ≈ compute_cost(cost2, time_range, x_at_tt, us_at_tt)
    end

    @test evaluate(cost1, xs, us) ≈ evaluate(cost2, xs, us)

end

@testset "TestQuadraticCost" begin
    horizon = 10
    dt = 1.
    times = dt * (cumsum(ones(horizon)) .- 1.)

    si = SystemInfo(2, 4, [2, 2])
    time_range = (0., horizon)
    
    # For a convex quadratic cost, make a second-order Taylor approximation of the cost about an arbitrary state-control
    # waypoint. Then revert the change using the negative of the state-ctrl waypoint and expect to recover the original
    # quadratic cost.
    @testset "UndoingQuadraticizationOfConvexQuadraticCostReturnsSameQuadratic" begin

        # Generate arbitrary quadratic costs centered around zero.
        all_quad_costs = generate_random_quadratic_costs(si; include_cross_costs=true, make_affine=true)

        # Generalize the tests by ensuring the initial state/control points are nonzero.
        init_x, init_us = generate_random_state_ctrl_point(si)
        add_offsets!(all_quad_costs[1], init_x, init_us)
        add_offsets!(all_quad_costs[2], init_x, init_us)
        x0, u0s = generate_random_state_ctrl_point(si)

        for ii in 1:num_agents(si)
            quad_cost = all_quad_costs[ii]

            approx_quad_cost = quadraticize_costs(quad_cost, time_range, x0, u0s)
            inverted_approx_quad_cost = quadraticize_costs(approx_quad_cost, time_range, init_x, init_us)
            
            # After inverting the quadraticization about x0, u0s, we should recover the initial quadratic cost.
            c1 = quad_cost
            c2 = inverted_approx_quad_cost

            # Rather than comparing directly, ensure that the cost mass of the constants is identical.
            constant_cost_mass_c1 = get_constant_state_cost_term(c1)
            constant_cost_mass_c2 = get_constant_state_cost_term(c2)

            @test get_quadratic_state_cost_term(c1) == get_quadratic_state_cost_term(c2)
            @test get_linear_state_cost_term(c1) ≈ get_linear_state_cost_term(c2)
            for ii in 1:num_agents(si)
                @test get_quadratic_control_cost_term(c1, ii) == get_quadratic_control_cost_term(c2, ii)
                @test get_linear_control_cost_term(c1, ii)  ≈ get_linear_control_cost_term(c2, ii)
                constant_cost_mass_c1 += get_constant_control_cost_term(c1, ii)
                constant_cost_mass_c2 += get_constant_control_cost_term(c2, ii)
            end

            # As described above, ensure that the cost mass of the constants is identical.
            @test constant_cost_mass_c1 ≈ constant_cost_mass_c2

            # Check that all costs are consistent.
            xs, us = generate_random_infeasible_trajectory(si, x0, u0s, horizon)
            ensure_same_costs(c1, approx_quad_cost, horizon, time_range, xs, us)
            ensure_same_costs(approx_quad_cost, c2, horizon, time_range, xs, us)
            ensure_same_costs(c1, c2, horizon, time_range, xs, us)
        end
    end

    # For a convex quadratic cost, make a second-order Taylor approximation of the cost about an arbitrary state-control
    # waypoint. We expect that computing cost at each time and over the whole trajectory results in identical values.
    # Repeat this process with randomly state-ctrl waypoints on the initial cost. 
    @testset "ApproximatingConvexQuadraticCostToSecondOrderEvaluatesIdenticalCosts" begin
        NUM_TESTS = 3
        
        all_quad_costs = generate_random_quadratic_costs(si; include_cross_costs=true, make_affine=true)

        for ii in 1:num_agents(si)
            quad_cost = all_quad_costs[ii]
            for n in 1:NUM_TESTS
                x0, u0s = generate_random_state_ctrl_point(si)
                approx_quad_cost = quadraticize_costs(quad_cost, time_range, x0, u0s)
                xs, us = generate_random_infeasible_trajectory(si, x0, u0s, horizon)
                
                ensure_same_costs(quad_cost, approx_quad_cost, horizon, time_range, xs, us)
            end
        end
    end

    # For a convex quadratic cost, make a second-order Taylor approximation of the cost about an arbitrary state-control
    # waypoint. We expect that computing cost at each time and over the whole trajectory results in identical values.
    # Repeat this process with randomly state-ctrl waypoints on the initial cost. We use the previous quadratic cost as
    # the starting cost for the next iteration of the test.
    @testset "IterativelyApproximatingConvexQuadraticCostToSecondOrderEvaluatesIdenticalCosts" begin
        NUM_TESTS = 3
        
        all_quad_costs = generate_random_quadratic_costs(si; include_cross_costs=true, make_affine=true)

        for ii in 1:num_agents(si)
            quad_cost = all_quad_costs[ii]
            for n in 1:NUM_TESTS
                x0, u0s = generate_random_state_ctrl_point(si)
                approx_quad_cost = quadraticize_costs(quad_cost, time_range, x0, u0s)
                xs, us = generate_random_infeasible_trajectory(si, x0, u0s, horizon)
                
                ensure_same_costs(quad_cost, approx_quad_cost, horizon, time_range, xs, us)
                all_quad_costs[ii] = approx_quad_cost
            end
        end
    end
end
