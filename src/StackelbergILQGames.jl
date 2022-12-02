# An algorithm that implements ILQGames for non-LQ Stackelberg games.

struct StackelbergControlStrategy
    T
    Ss
    Ls
end

function strategy_distance(strategy1::StackelbergControlStrategy, strategy2::StackelbergControlStrategy)

    # Normalize so we avoid numerical issues. Q: Or does this introduce new numerical issues?
    S1s = normalize.(strategy1.Ss)
    L1s = normalize.(strategy1.Ls)
    S2s = normalize.(strategy2.Ss)
    L2s = normalize.(strategy2.Ls)

    return sum(norm.(S1s - S2s) + norm.(L1s - L2s))
end

function stackelberg_2p_ilqgames_iteration(leader_idx::Int,
                                           dyn::Dynamics,
                                           costs::AbstractVector{<:Cost},
                                           x0,
                                           t0,
                                           current_ctrl_strats::StackelbergControlStrategy,
                                           x_refs::AbstractArray{Float64},
                                           u_refs::AbstractVector{<:AbstractArray{Float64}})
    N = num_agents(dyn)
    T = current_ctrl_strats.T

    # 1. Propagate the current control strategy forward in time using the nonlinear dynamics to get the state and
    #    control trajectories.
    x_hats, u_hats = unroll_feedback(dyn, current_ctrl_strats.Ss, x0)

    # 2. Compute errors relative to reference trajectories.
    dxs = x_refs - x_hats
    dus = u_refs - u_hats

    # 3. Create an LQ approximation of the game about the error state and controls.
    # 4. Solve an LQ Stackelberg game given the approximated LQ conditions.
    Ss, Ls = solve_approximated_lq_stackelberg_feedback(dyn, costs, T, t0, x_hats, u_hats, leader_idx)

    # 5. Adjust the control strategies based on the approximate solution.
    new_control_strats = StackelbergControlStrategy(T, Ss, Ls)

    for tt in 1:T
        for ii in 1:N
            u_hats[ii][:, tt] -= Ss[ii][:, :, tt] * dxs[:, tt]
        end
    end

    return new_control_strats, u_hats

end

THRESHOLD = 0.01
function stackelberg_ilqgames(leader_idx::Int,
                              T::Int,
                              x0,
                              t0,
                              dyn::Dynamics,
                              costs::AbstractVector{QuadraticCost},
                              initial_ctrl_strats::StackelbergControlStrategy,
                              x_refs::AbstractArray{Float64},
                              u_refs::AbstractVector{<:AbstractArray{Float64}};
                              threshold::Float64 = THRESHOLD)
    MAX_ITERS = 100
    num_iters = 0

    N = num_agents(dyn)

    prev_ctrl_strats = initial_ctrl_strats
    current_ctrl_strats = initial_ctrl_strats
    is_converged = false

    function simple_iter(ctrl_strats::StackelbergControlStrategy, x_references, u_references)
        return stackelberg_2p_ilqgames_iteration(leader_idx, dyn, costs, x0, t0, current_ctrl_strats, x_references, u_references)
    end

    while !is_converged && num_iters < MAX_ITERS
        prev_ctrl_strats = current_ctrl_strats

        # Q: What do I do with u_hat here?
        current_ctrl_strats, u_hats = simple_iter(current_ctrl_strats, x_refs, u_refs)

        is_converged = !(strategy_distance(current_ctrl_strats, prev_ctrl_strats) > threshold)
        num_iters += 1
    end

    return current_ctrl_strats, is_converged, num_iters
end

export StackelbergControlStrategy, stackelberg_ilqgames
