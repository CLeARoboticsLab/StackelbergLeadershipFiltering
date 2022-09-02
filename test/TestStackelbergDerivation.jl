# unit tests for confirming my derivation matches Basar's derivation over one time step with random inputs
using LinearAlgebra
using Random: seed!
using Test: @test, @testset

seed!(0)

function compute_basar_recursion_one_step(A, B1, B2, Q1, Q2, L1ₜ₊₁, L2ₜ₊₁, R11, R22, R12, R21, S1ₜ=nothing, S2ₜ=nothing)
    # Computes the basar recursion at a single timestep.
    inner_inv_1 = (I + B2 * B2' * L2ₜ₊₁) \ B1
    inner_inv_2 = (I + B2' * L2ₜ₊₁ * B2) \ B2'
    inner_inv_3 = (I + L2ₜ₊₁ * B2 * B2') \ L1ₜ₊₁

    outer_lhs = (inner_inv_1' * L1ₜ₊₁ * inner_inv_1
           + B1' * L2ₜ₊₁ * B2 * ((I + B2' * L2ₜ₊₁ * B2) \ R12) * ((I + B2' * L2ₜ₊₁ * B2) \ (B2' * L2ₜ₊₁ * B1)) + I)
    outer_rhs = B1' * (inner_inv_3 * inv(I + B2 * B2' * L2ₜ₊₁)
                  + L2ₜ₊₁ * inner_inv_2' * R12 * inner_inv_2 * L2ₜ₊₁) * A
    S1ₜ = outer_lhs \ outer_rhs

    S2ₜ = ((I + B2' * L2ₜ₊₁ * B2) \ (B2' * L2ₜ₊₁ * (A - B1 * S1ₜ)))

    dynamics_tp1 = A - B1 * S1ₜ - B2 * S2ₜ
    L1 = dynamics_tp1' * L1ₜ₊₁ * dynamics_tp1 + S1ₜ' * S1ₜ + S2ₜ' * R12 * S2ₜ + Q1
    L2 = dynamics_tp1' * L2ₜ₊₁ * dynamics_tp1 + S1ₜ' * R21 * S1ₜ + S2ₜ' * S2ₜ + Q2
    return [S1ₜ, S2ₜ, L1, L2]
end

function compute_hamzah_derivation_one_step(A, B1, B2, Q1, Q2, L1ₜ₊₁, L2ₜ₊₁, R11, R22, R12, R21, S1ₜ=nothing, S2ₜ=nothing)
    # Computes the recursion at a single time using hamzah's derivation.
    Dₜ = (R22 + B2' * L2ₜ₊₁ * B2) \ (B2' * L2ₜ₊₁)

    lhs = R11 + B1' * Dₜ' * R12 * Dₜ * B1 + (B1 - B2 * Dₜ * B1)' * L1ₜ₊₁ * (B1 - B2 * Dₜ * B1)
    rhs = ((B1' * Dₜ' * R12 * Dₜ) + (B1 - B2 * Dₜ * B1)' * L1ₜ₊₁ * (I - B2 * Dₜ)) * A

    S1ₜ = lhs \ rhs
    S2ₜ = Dₜ * (A - B1 * S1ₜ)

    dynamics_tp1 = A - B1 * S1ₜ - B2 * S2ₜ
    L1 = Q1 + S1ₜ' * R11 * S1ₜ + S2ₜ' * R12 * S2ₜ + dynamics_tp1' * L1ₜ₊₁ * dynamics_tp1
    L2 = Q2 + S1ₜ' * R21 * S1ₜ + S2ₜ' * R22 * S2ₜ + dynamics_tp1' * L2ₜ₊₁ * dynamics_tp1
    return [S1ₜ, S2ₜ, L1, L2]
end

function make_symmetric_pos_def_matrix(dim)
    # Generate a matrix of the specified size.
    A = rand(dim, dim)

    # Make it symmetric.
    A = (1/2) * (A + A')

    # Make it positive definite.
    smallest_eigenval = minimum(eigvals(A))
    if smallest_eigenval <= 0
        # Shift the eigenvalues by a factor of the absolute value of the smallest eigenvalue.
        A += (2 - rand()) * abs(smallest_eigenval) * I
    end
    return A
end

@testset "TestLQStackelbergDerivationOneLine" begin
    stackelberg_leader_idx = 1

    nx = 2
    nu = 2

    A = make_symmetric_pos_def_matrix(nx)
    B1 = rand(nx, nu)
    B2 = rand(nx, nu)

    L1ₜ₊₁ = make_symmetric_pos_def_matrix(nx)
    L2ₜ₊₁ = make_symmetric_pos_def_matrix(nx)

    Q1 = make_symmetric_pos_def_matrix(nx)
    Q2 = make_symmetric_pos_def_matrix(nx)

    R11 = I
    R12 = make_symmetric_pos_def_matrix(nu)
    R21 = make_symmetric_pos_def_matrix(nu)
    R22 = I

    S1ₜ = nothing
    S2ₜ = nothing

    basar = compute_basar_recursion_one_step(A, B1, B2, Q1, Q2, L1ₜ₊₁, L2ₜ₊₁, R11, R22, R12, R21, S1ₜ, S2ₜ)
    hamzah = compute_hamzah_derivation_one_step(A, B1, B2, Q1, Q2, L1ₜ₊₁, L2ₜ₊₁, R11, R22, R12, R21, S1ₜ, S2ₜ)

    # Validate that L_i matrices are symmetric and positive definite.
    L1_basar = basar[3]
    L2_basar = basar[4]
    @test L1_basar ≈ L1_basar'
    @test L2_basar ≈ L2_basar'
    @test minimum(eigvals(L1_basar)) > 0
    @test minimum(eigvals(L2_basar)) > 0

    L1_hamzah = hamzah[3]
    L2_hamzah = hamzah[4]
    @test L1_hamzah ≈ L1_hamzah'
    @test L2_hamzah ≈ L2_hamzah'
    @test minimum(eigvals(L1_hamzah)) > 0
    @test minimum(eigvals(L2_hamzah)) > 0

    # Validate that S_i, L_i matrices are the same between methods.
    @test basar[1] ≈ hamzah[1]
    @test basar[2] ≈ hamzah[2]
    @test basar[3] ≈ hamzah[3]
    @test basar[4] ≈ hamzah[4]
end
