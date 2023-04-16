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

end