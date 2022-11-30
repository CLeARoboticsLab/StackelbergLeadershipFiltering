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
                                           costs::AbstractArray{Cost}
                                           x0,
                                           t0,
                                           current_ctrl_strats::StackelbergControlStrategy;
                                           x_refs=zeros(zeros(xdim(dyn), horizon)),
                                           u_refs=[zeros(udim(dyn, ii), horizon) for ii in 1:num_agents(dyn)])
    N = num_agents(dyn)
    T = current_ctrl_strats.T

    # 1. Propagate the current control strategy forward in time using the nonlinear dynamics to get the state and
    #    control trajectories.
    x_hats, u_hats = unroll_feedback(dyn, current_ctrl_strats.Ss, x0)

    # 2. Compute errors relative to reference trajectories.
    dxs = x_refs - x_hats
    dus = u_refs - u_hats

    # 3. Create an LQ approximation of the game about the error state and controls.
    lin_dyns = Array{AbstractArray{Float64}}(undef, T)
    quad_costs = Array{AbstractArray{Float64}}(undef, T)

    for tt in 1:T
        us_at_tt = [us[ii][tt, :] for ii in 1:N]
        lin_dyns[tt] = linearize_dynamics(dyn, tt+t0, x_hats[t, :], us_at_tt)
        quad_costs[tt] = quadraticize_costs(costs, tt+t0, x_hats[t, :], us_at_tt)
    end

    # 4. Solve an LQ Stackelberg game given the approximated LQ conditions.
    # TODO: Make the nash and stackelberg solvers work with time varying systems.
    Ss, Ls = solve_lq_stackelberg_feedback(lin_dyns, quad_costs, T, leader_idx)

    # 5. Adjust the control strategies based on the approximate solution.
    # TODO
    new_control_strats = StackelbergControlStrategy(u_hats - current_ctrl_strats.Ss * dxs)

    return new_control_strats

end

function stackelberg_ilqgames(leader_idx::Int,
                              T,
                              x0,
                              t0,
                              dyn::Dynamics,
                              costs::AbstractArray{QuadraticCost},
                              initial_ctrl_strats::StackelbergControlStrategy;
                              x_refs=zeros(zeros(xdim(dyn), horizon)),
                              u_refs=[zeros(udim(dyn, ii), horizon) for ii in 1:num_agents(dyn)])
    THRESHOLD = 0.01
    MAX_ITERS = 100
    iters = 0

    prev_ctrl_strats = initial_ctrl_strats
    current_ctrl_strats = initial_ctrl_strats

    # x_refs = zeros(zeros(xdim(dyn), horizon))
    # u_refs = [zeros(udim(dyn, ii), horizon) for ii in 1:num_agents(dyn)]

    function simple_iter(ctrl_strats::StackelbergControlStrategy, x_references, u_references)
        return stackelberg_2p_ilqgames_iteration(leader_idx, dyn, costs, x0, t0, T, current_ctrl_strats, x_references, u_references)
    end

    while strategy_distance(current_ctrl_strats, prev_ctrl_strats) > THRESHOLD && iters < MAX_ITERS
        prev_ctrl_strats = current_ctrl_strats
        current_ctrl_strats = simple_iter(current_ctrl_strats, x_refs, u_refs)
    end

    exceeded_max_iters = iters >= MAX_ITERS
    is_converged = !exceeded_max_iters
    return current_ctrl_strats, is_converged
end
