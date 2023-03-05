using LinearAlgebra
using Random
using StackelbergControlHypothesesFiltering

# Generates a random square matrix that is symmetric and positive definite.
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

function generate_random_linear_dynamics(sys_info::SystemInfo)
    N = num_agents(sys_info)
    num_states = xdim(sys_info)

    A = rand(num_states, num_states) 
    Bs = [rand(num_states, udim(sys_info, ii)) for ii in 1:N]

    return LinearDynamics(A, Bs, sys_info)
end

function generate_random_quadratic_costs(sys_info::SystemInfo; include_cross_costs=true, make_affine=false)
    N = num_agents(sys_info)
    num_states = xdim(sys_info)

    # Make state cost.
    Q = make_symmetric_pos_def_matrix(num_states)
    Q = (make_affine) ? vcat(hcat(Q, zeros(num_states)), hcat(zeros(1, num_states), 1.)) : Q
    costs = [QuadraticCost(Q) for _ in 1:N]


    # Make control costs.
    for ii in 1:N
        for jj in 1:N
            # Generate the control costs, factoring in the flag for cross control.
            num_uj = udim(sys_info, jj)
            if ii != jj && !include_cross_costs
                R_ij = zeros(num_uj, num_uj)
                add_control_cost!(costs[ii], jj, R_ij)
                continue
            end
            R_ij = make_symmetric_pos_def_matrix(num_uj)
            R_ij = (make_affine) ? vcat(hcat(R_ij, zeros(num_uj)), hcat(zeros(1, num_uj), 1.)) : R_ij
            add_control_cost!(costs[ii], jj, R_ij)
        end
    end

    return costs
end

export make_symmetric_pos_def_matrix, generate_random_linear_dynamics, generate_random_quadratic_costs
