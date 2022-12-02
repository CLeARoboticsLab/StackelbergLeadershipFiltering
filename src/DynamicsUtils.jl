using SparseArrays

# Utilities for managing linear and nonlinear dynamics.

# Every Dynamics is assumed to have the following functions defined on it:
# - linearize_dynamics(dyn, x, us) - this function linearizes the dynamics given the state and controls.
# - propagate_dynamics(cost, t, x, us) - this function propagates the dynamics to the next timestep.
# Every Dynamics struct must have a sys_info field of type SystemInfo.
abstract type Dynamics end

# A type that every nonlinear dynamics struct (unique per use case) can inherit from. These need to have the same
# functions as the Dynamics type.
abstract type NonlinearDynamics <: Dynamics end

# TODO(hamzah) Add better tests for the LinearDynamics struct and associated functions.
struct LinearDynamics <: Dynamics
    A  # state
    Bs # controls
    sys_info::SystemInfo
end
# Constructor for linear dynamics that auto-generates the system info.
LinearDynamics(A, Bs) = LinearDynamics(A, Bs, SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)]))

function propagate_dynamics(dyn::LinearDynamics, t, x, us)
    N = dyn.sys_info.num_agents
    x_next = dyn.A * x
    for i in 1:N
        ui = reshape(us[i], dyn.sys_info.num_us[i], 1)
        x_next += dyn.Bs[i] * ui
    end
    return x_next
end

function linearize_dynamics(dyn::LinearDynamics, t, x, us)
    return dyn
end

# This util is meant to be used where the state can be decomposed into N players which each act according to unicycle
# dynamics. A unicycle dynamics model with one actor has a 4-element state including 2D-position, angle, and speed.
# There are two controls, turn rate and acceleration.
# X_i = [px_i py_i theta_i v_i]'
# U_i = [omega_i alpha_i]'
# Unicycle dynamics indepedent of other actors can then be composed into a single state by concatenation and with some
# minor changes to the control matrices B_i.
struct UnicycleDynamics <: NonlinearDynamics
    sys_info::SystemInfo
end

# argument N - number of players
UnicycleDynamics(N) = UnicycleDynamics(SystemInfo(N, 4*N, [2 for i in 1:N]))

function propagate_dynamics(dyn::UnicycleDynamics, t, x, us)
    N = dyn.sys_info.num_agents
    @assert N == length(us)
    @assert xdim(dyn) == 4 * N
    @assert udim(dyn) == 2 * N

    x_dot = zeros(xdim(dyn))

    for ii in 1:N
        start_idx = 4 * (ii-1)
        px = x[start_idx + 1]
        py = x[start_idx + 2]
        theta = x[start_idx + 3]
        v = x[start_idx + 4]

        turn_rate = us[ii][1]
        accel = us[ii][2]

        x_dot[start_idx+1:start_idx+4] = [v * cos(theta); v * sin(theta); turn_rate; accel]
    end

    return x_dot
end

function linearize_dynamics(dyn::UnicycleDynamics, t, x, us)
    N = dyn.sys_info.num_agents
    @assert N == length(us)
    @assert xdim(dyn) == 4 * N

    As = [sparse(zeros(4, 4)) for ii in 1:N]
    Bs = [zeros(xdim(dyn), 2) for ii in 1:N]

    for ii in 1:N
        @assert udim(dyn, ii) == 2

        start_idx = 4 * (ii-1)
        theta = x[start_idx + 3]
        v = x[start_idx + 4]

        # Compute the state and controls for each actor.
        s = sin(theta)
        c = cos(theta)
        As[ii][1:2, 3:4] = [-v*s c; v*c s]
        Bs[ii][start_idx+3:start_idx+4, 1:2] = [1 0; 0 1]
    end
    # Combine the As into one large A matrix and add in the zeroth order term of the Taylor expansion.
    A = I + Matrix(blockdiag(As...))

    return LinearDynamics(A, Bs, dyn.sys_info)
end


# Export the types of dynamics.
export Dynamics, NonlinearDynamics, LinearDynamics, UnicycleDynamics

# Export the functionality each Dynamics requires.
export propagate_dynamics, linearize_dynamics


# Dimensionality helpers.
function num_agents(dyn::Dynamics)
    return num_agents(dyn.sys_info)
end

function xdim(dyn::Dynamics)
    return xdim(dyn.sys_info)
end

function udim(dyn::Dynamics)
    return udim(dyn.sys_info)
end

function udim(dyn::Dynamics, player_idx)
    return udim(dyn.sys_info, player_idx)
end

export num_agents, xdim, udim


# TODO(hamzah) Add better tests for the unroll_feedback, unroll_raw_controls functions.
# TODO(hamzah) Abstract the unroll_feedback, unroll_raw_controls functions to not assume linear feedback P.

# Function to unroll a set of feedback matrices from an initial condition.
# Output is a sequence of states xs[:, time] and controls us[player][:, time].
export unroll_feedback
function unroll_feedback(dyn::Dynamics, Ps, x₁)
    @assert length(x₁) == xdim(dyn)

    N = length(Ps)
    @assert N == dyn.sys_info.num_agents

    horizon = last(size(first(Ps)))

    # Populate state/control trajectory.
    xs = zeros(xdim(dyn), horizon)
    xs[:, 1] = x₁
    us = [zeros(udim(dyn, ii), horizon) for ii in 1:N]
    for tt in 2:horizon
        for ii in 1:N
            us[ii][:, tt - 1] = -Ps[ii][:, :, tt - 1] * xs[:, tt - 1]
        end

        us_prev = [us[i][:, tt-1] for i in 1:N]
        xs[:, tt] = propagate_dynamics(dyn, tt, xs[:, tt-1], us_prev)
    end

    # Controls at final time.
    for ii in 1:N
        us[ii][:, horizon] = -Ps[ii][:, :, horizon] * xs[:, horizon]
    end

    return xs, us
end

# As above, but replacing feedback matrices `P` with raw control inputs `u`.
export unroll_raw_controls
function unroll_raw_controls(dyn::Dynamics, us, x₁)
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
        xs[:, tt] = propagate_dynamics(dyn, tt, xs[:, tt-1], us_prev)
    end

    return xs
end
