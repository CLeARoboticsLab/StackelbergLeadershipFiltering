# TODO(hamzah) Add better tests for the unroll_feedback, unroll_raw_controls functions.

# A MultiplayerControlStrategy must have the following fields
# - num_players: number of players, N
# - T:           horizon
# - Ps:          an N-length vector of feedback gains across times 1:T
# - Zs:          an N-length vector of future state costs across times 1:T
# It must also have a function defined as follows:
# - apply_control_strategy: accepts a control strategy, a time, and a state, and produced the strategy at that time.
abstract type MultiplayerControlStrategy end

# Function to unroll a set of feedback matrices from an initial condition.
# Output is a sequence of states xs[:, time] and controls us[player][:, time].
function unroll_feedback(dyn::Dynamics, times::AbstractVector{Float64}, control_strategy::MultiplayerControlStrategy, x₁)
    @assert length(x₁) == xdim(dyn)

    N = control_strategy.num_players
    @assert N == num_agents(dyn)

    horizon = control_strategy.horizon

    # Populate state/control trajectory.
    xs = zeros(xdim(dyn), horizon)
    xs[:, 1] = x₁
    us = [zeros(udim(dyn, ii), horizon) for ii in 1:N]
    for tt in 2:horizon
        ctrls_at_ttm1 = apply_control_strategy(tt-1, control_strategy, xs[:, tt-1])
        for ii in 1:N
            us[ii][:, tt - 1] = ctrls_at_ttm1[ii]
        end

        us_prev = [us[i][:, tt-1] for i in 1:N]
        time_range = (times[tt-1], times[tt])
        xs[:, tt] = propagate_dynamics(dyn, time_range, xs[:, tt-1], us_prev)
    end

    # Controls at final time.
    final_ctrls = apply_control_strategy(horizon, control_strategy, xs[:, horizon])
    for ii in 1:N
        us[ii][:, horizon] = final_ctrls[ii]
    end

    return xs, us
end

# As above, but replacing feedback matrices `P` with raw control inputs `u`.
function unroll_raw_controls(dyn::Dynamics, times::AbstractVector{Float64}, us, x₁)
    @assert length(x₁) == xdim(dyn)

    N = length(us)
    @assert N == dyn.sys_info.num_agents

    horizon = last(size(first(us)))

    # Populate state trajectory.
    xs = zeros(xdim(dyn), horizon)
    xs[:, 1] = x₁
    us = [zeros(udim(dyn, ii), horizon) for ii in 1:N]
    for tt in 2:horizon
        us_prev = [us[i][:, tt-1] for i in 1:N]
        time_range = (times[tt-1], times[tt])
        xs[:, tt] = propagate_dynamics(dyn, time_range, xs[:, tt-1], us_prev)
    end

    return xs
end

# Export the abstract type.
export MultiplayerControlStrategy, unroll_feedback, unroll_raw_controls
