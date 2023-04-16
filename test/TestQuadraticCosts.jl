# Unit tests for costs.
using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Random: seed!
using Test: @test, @testset
include("TestUtils.jl")

seed!(0)

# Ensure that the quadratic cost and quadratic cost with offset at 0 are identical in vari
@testset "TestQuadraticCosts" begin
    x_dim = 3
    u_dims = [1, 2, 3]
    num_players = 3
    num_tests = 2

    time_range = (0., rand())


    @testset "QuadraticizingQuadraticCostIsIdenticalToOriginalQuadraticCost" begin
        Q = make_symmetric_pos_def_matrix(x_dim)
        q = rand(x_dim)
        cq = rand()
        cost = QuadraticCost(Q, q, cq)
        for ii in 1:num_players
            u_dim = u_dims[ii]
            R = make_symmetric_pos_def_matrix(u_dim)
            r = rand(u_dim)
            cr = rand()
            add_control_cost!(cost, ii, R; r, cr)
        end

        x0 = rand(x_dim)
        u0s = [zeros(u_dims[ii]) for ii in 1:num_players]
        approximated_cost = quadraticize_costs(cost, time_range, x0, u0s)
        
        # Generate random inputs.
        x = rand(x_dim)
        us = [rand(u_dims[ii]) for ii in 1:num_players]

        # Computation should be identical because quadratic approximation a quadratic function is exact.
        @test compute_cost(cost, time_range, x, us) ≈ compute_cost(approximated_cost, time_range, x, us)

        # Derivatives should also be identical.
        @test Gx(cost, time_range, x, us) == Gx(approximated_cost, time_range, x, us)
        @test Gxx(cost, time_range, x, us) == Gxx(approximated_cost, time_range, x, us)

        cost_Dus = Gus(cost, time_range, x, us)
        approx_Dus = Gus(approximated_cost, time_range, x, us)
        cost_Duus = Guus(cost, time_range, x, us)
        approx_Duus = Guus(approximated_cost, time_range, x, us)
        for ii in 1:num_players
            @test cost_Dus[ii] == approx_Dus[ii]
            @test cost_Duus[ii] == approx_Duus[ii]
        end
    end


    @testset "TranslatingQuadraticCostIsIdenticalWithShiftedArgument" begin
        Q = make_symmetric_pos_def_matrix(x_dim)
        q = rand(x_dim)
        cq = rand()
        cost = QuadraticCost(Q, q, cq)
        for ii in 1:num_players
            u_dim = u_dims[ii]
            R = make_symmetric_pos_def_matrix(u_dim)
            r = rand(u_dim)
            cr = rand()
            add_control_cost!(cost, ii, R; r, cr)
        end

        x0 = rand(x_dim)
        u0s = [zeros(u_dims[ii]) for ii in 1:num_players]
        translated_cost = translate_quadratic_cost(cost, x0, u0s)
        
        # Generate random inputs.
        x = rand(x_dim)
        us = [rand(u_dims[ii]) for ii in 1:num_players]

        # Computation should be identical under shift.
        dx = x - x0
        dus = [us[ii] - u0s[ii] for ii in 1:num_players]
        @test compute_cost(cost, time_range, dx, dus) == compute_cost(translated_cost, time_range, x, us)

        # Derivatives should also be identical under shift.
        @test Gx(cost, time_range, dx, dus) ≈ Gx(translated_cost, time_range, x, us)
        @test Gxx(cost, time_range, dx, dus) ≈ Gxx(translated_cost, time_range, x, us)

        shifted_Dus = Gus(cost, time_range, dx, dus)
        translated_Dus = Gus(translated_cost, time_range, x, us)
        shifted_Duus = Guus(cost, time_range, dx, dus)
        translated_Duus = Guus(translated_cost, time_range, x, us)
        for ii in 1:num_players
            @test shifted_Dus[ii] == translated_Dus[ii]
            @test shifted_Duus[ii] == translated_Duus[ii]
        end
    end


    @testset "TranslatedQuadraticCostIsIdenticalWithInverseShiftedArgument" begin
        Q = make_symmetric_pos_def_matrix(x_dim)
        q = rand(x_dim)
        cq = rand()
        cost = QuadraticCost(Q, q, cq)
        for ii in 1:num_players
            u_dim = u_dims[ii]
            R = make_symmetric_pos_def_matrix(u_dim)
            r = rand(u_dim)
            cr = rand()
            add_control_cost!(cost, ii, R; r, cr)
        end

        x0 = rand(x_dim)
        u0s = [zeros(u_dims[ii]) for ii in 1:num_players]
        translated_cost = translate_quadratic_cost(cost, x0, u0s)
        
        # Generate random inputs.
        x = rand(x_dim)
        us = [rand(u_dims[ii]) for ii in 1:num_players]

        # Computation should be identical under shift.
        inv_dx = x + x0
        inv_dus = [us[ii] + u0s[ii] for ii in 1:num_players]
        @test compute_cost(cost, time_range, x, us) ≈ compute_cost(translated_cost, time_range, inv_dx, inv_dus)

        # Derivatives should also be identical, under inverse shift.
        @test Gx(cost, time_range, x, us) ≈ Gx(translated_cost, time_range, inv_dx, inv_dus)
        @test Gxx(cost, time_range, x, us) ≈ Gxx(translated_cost, time_range, inv_dx, inv_dus)

        cost_Dus = Gus(cost, time_range, x, us)
        translated_inv_shifted_Dus = Gus(translated_cost, time_range, inv_dx, inv_dus)
        cost_Duus = Guus(cost, time_range, x, us)
        translated_inv_shifted_Duus = Guus(translated_cost, time_range, inv_dx, inv_dus)
        for ii in 1:num_players
            @test cost_Dus[ii] == translated_inv_shifted_Dus[ii]
            @test cost_Duus[ii] == translated_inv_shifted_Duus[ii]
        end
    end


    # @testset "QuadraticizingQuadCostWithOffsetIsIdenticalToOriginalQuadraticCostWithOffset" begin
    #     Q = make_symmetric_pos_def_matrix(x_dim)
    #     q = rand(x_dim)
    #     cq = rand()
    #     q_cost = QuadraticCost(Q, q, cq)
    #     for ii in 1:num_players
    #         u_dim = u_dims[ii]
    #         R = make_symmetric_pos_def_matrix(u_dim)
    #         r = rand(u_dim)
    #         cr = rand()
    #         add_control_cost!(q_cost, ii, R; r, cr)
    #     end

    #     # Offset the cost by a in state, bs in ctrls.
    #     a = rand(x_dim)
    #     bs = [rand(u_dims[ii]) for ii in 1:num_players]
    #     quad_cost_w_offset = QuadraticCostWithOffset(q_cost, a, bs)

    #     # Quadraticize about arbitrary state and controls.
    #     x0 = rand(x_dim)
    #     u0s = [rand(u_dims[ii]) for ii in 1:num_players]
    #     approximated_cost = quadraticize_costs(quad_cost_w_offset, time_range, x0, u0s)

    #     # Generate arbitrary inputs.
    #     x = rand(x_dim)
    #     us = [rand(u_dims[ii]) for ii in 1:num_players]

    #     # Computation should be identical when evaluated at arbitrary x, us because quadratic approximation a quadratic function is exact.
    #     @test compute_cost(quad_cost_w_offset, time_range, x, us) == compute_cost(approximated_cost, time_range, x, us)
    #     println("cost difference: ", compute_cost(quad_cost_w_offset, time_range, x, us) - compute_cost(approximated_cost, time_range, x, us))
    #     println("cq: ", quad_cost_w_offset.q_cost.cq, " ", approximated_cost.cq)
    #     println("crqs: ", quad_cost_w_offset.q_cost.crs, " ", approximated_cost.crs)


    #     # Derivatives should also be identical.
    #     @test Gx(quad_cost_w_offset, time_range, x, us) ≈ Gx(approximated_cost, time_range, x, us)
    #     @test Gxx(quad_cost_w_offset, time_range, x, us) ≈ Gxx(approximated_cost, time_range, x, us)

    #     cost_Dus = Gus(quad_cost_w_offset, time_range, x, us)
    #     approx_Dus = Gus(approximated_cost, time_range, x, us)
    #     cost_Duus = Guus(quad_cost_w_offset, time_range, x, us)
    #     approx_Duus = Guus(approximated_cost, time_range, x, us)
    #     for ii in 1:num_players
    #         @test cost_Dus[ii] ≈ approx_Dus[ii]
    #         @test cost_Duus[ii] ≈ approx_Duus[ii]
    #     end
    # end


    # @testset "TranslatedQuadraticCostEquivalentToQuadraticizingQuadCostWithOffset" begin
    #     Q = make_symmetric_pos_def_matrix(x_dim)
    #     q = rand(x_dim)
    #     cq = rand()
    #     c1 = QuadraticCost(Q, q, cq)
    #     for ii in 1:num_players
    #         u_dim = u_dims[ii]
    #         R = make_symmetric_pos_def_matrix(u_dim)
    #         r = rand(u_dim)
    #         cr = rand()
    #         add_control_cost!(c1, ii, R; r, cr)
    #     end

    #     x0 = rand(x_dim)
    #     u0s = [zeros(u_dims[ii]) for ii in 1:num_players]
    #     translated_cost = translate_quadratic_cost(c1, x0, u0s)

    #     cost_w_offset = QuadraticCostWithOffset(c1, x0, u0s)


    #     x = rand(x_dim)
    #     us = [rand(u_dims[ii]) for ii in 1:num_players]

    #     println("compute costs (translate, offset): ", compute_cost(translated_cost, time_range, x, us), " ", compute_cost(cost_w_offset, time_range, x, us))

    #     @assert all(cost_w_offset.Q .== translated_cost.Q)
    #     println("q: ", cost_w_offset.q, "\n   ", translated_cost.q)
    #     @assert all(cost_w_offset.q .== translated_cost.q)
    #     println(cost_w_offset.cq, " ", translated_cost.cq)
    #     @assert all(cost_w_offset.cq .== translated_cost.cq)


    #     x_new = rand(x_dim)
    #     quadraticized_translation_cost = quadraticize_costs(translated_cost, time_range, x_new, u0s)
    #     quadraticized_offset_cost = quadraticize_costs(QuadraticCostWithOffset(c1, x0, u0s), time_range, x_new, u0s)

    # end

    # @testset "QuadraticCostWithOffsetFunctionOutputsMatchQuadratic" begin
    #     Q = make_symmetric_pos_def_matrix(x_dim)
    #     q = rand(x_dim)
    #     cq = rand()
    #     c1 = QuadraticCost(Q, q, cq)
    #     for ii in 1:num_players
    #         u_dim = u_dims[ii]
    #         R = make_symmetric_pos_def_matrix(u_dim)
    #         r = rand(u_dim)
    #         cr = rand()
    #         add_control_cost!(c1, ii, R; r, cr)
    #     end

    #     # Make the associated quadratic cost with offset.
    #     c2 = QuadraticCostWithOffset(c1)

    #     for tt in 1:num_tests
    #         x = rand(x_dim)
    #         us = [rand(u_dims[ii]) for ii in 1:length(u_dims)]

    #         @test compute_cost(c1, time_range, x, us) ≈ compute_cost(c2, time_range, x, us)
    #         @test Gx(c1, time_range, x, us) ≈ Gx(c2, time_range, x, us)
    #         @test all([Gus(c1, time_range, x, us)[ii] ≈ Gus(c2, time_range, x, us)[ii] for ii in 1:num_players])
    #         @test Gxx(c1, time_range, x, us) ≈ Gxx(c2, time_range, x, us)
    #         @test all([Guus(c1, time_range, x, us)[ii] ≈ Guus(c2, time_range, x, us)[ii] for ii in 1:num_players])

    #         quad1 = quadraticize_costs(c1, time_range, x, us)
    #         quad2 = quadraticize_costs(c2, time_range, x, us)
    #         @test get_quadratic_state_cost_term(quad1) ≈ get_quadratic_state_cost_term(quad2)
    #         @test get_linear_state_cost_term(quad1) ≈ get_linear_state_cost_term(quad2)
    #         @test get_constant_state_cost_term(quad1) ≈ get_constant_state_cost_term(quad2)
    #         @test all([get_quadratic_control_cost_term(quad1, ii) ≈ get_quadratic_control_cost_term(quad2, ii) for ii in 1:num_players])
    #         @test all([get_linear_control_cost_term(quad1, ii) ≈ get_linear_control_cost_term(quad2, ii) for ii in 1:num_players])
    #         @test all([get_constant_control_cost_term(quad1, ii) ≈ get_constant_control_cost_term(quad2, ii) for ii in 1:num_players])
    #     end
    # end

end