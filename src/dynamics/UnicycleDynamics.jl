using SparseArrays

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

# Constructor
UnicycleDynamics(num_players::Int) = UnicycleDynamics(SystemInfo(num_players, 4*num_players, 2*ones(num_players)))
UnicycleDynamics(num_players::Int, dt::Float64) = UnicycleDynamics(SystemInfo(num_players, 4*num_players, 2*ones(num_players), 0, dt))

export UnicycleDynamics

function dx(dyn::UnicycleDynamics,
            time_range,
            x::AbstractVector{TX},
            us::AbstractVector{<:AbstractVector{TU}},
            v) where {TX, TU}
    # TODO(hamzah) - propagate_dynamics not implemented with process noise for UnicycleDynamics".
    # TODO: Unicycle dynamics doesn't currently support process noise.
    @assert isnothing(v)

    N = num_agents(dyn)
    @assert N == length(us)
    @assert size(x, 1) == 4 * N
    for ii in 1:N
        @assert size(us[ii], 1) == 2
    end

    dx_t = zeros(TX, xdim(dyn))
    for ii in 1:N
        start_idx = 4 * (ii-1)
        px = x[start_idx + 1]
        py = x[start_idx + 2]
        theta = x[start_idx + 3]
        vel = x[start_idx + 4]

        turn_rate = us[ii][1]
        accel = us[ii][2]

        dx_t[start_idx+1:start_idx+4] = [vel * cos(theta); vel * sin(theta); turn_rate; accel]
    end
    return dx_t
end

# TODO(hamzah) - unify propagate dynamics into the DynamicsUtils, with separate validators and pre/post-process.
function propagate_dynamics(dyn::UnicycleDynamics,
                            time_range,
                            x,
                            us,
                            v)
    # TODO(hamzah) - propagate_dynamics not implemented with process noise for UnicycleDynamics".
    # TODO: Unicycle dynamics doesn't currently support process noise.
    @assert isnothing(v)

    N = num_agents(dyn)
    @assert N == length(us)
    @assert size(x, 1) == 4 * N
    for ii in 1:N
        @assert size(us[ii], 1) == 2
    end

    @assert !is_continuous(dyn) "Can only propagate discrete-time dynamics objects."

    dt = sampling_time(dyn)
    x_tp1 = x + dt * dx(dyn, time_range, x, us, v)

    # TODO(hamzah): Wrapping the angle is generally preferred, but causes issues with autodiff for some reason. Explore this.
    for ii in 1:N
        start_idx = 4 * (ii-1)
        # Wrap angle after propagation to bound in [-pi, pi).
        x_tp1[start_idx+3] = wrap_angle(x_tp1[start_idx+3])
    end

    return x_tp1
end

# These are the continuous derivatives of the unicycle dynamics with respect to x and u.

function Fx(dyn::UnicycleDynamics, time_range, x, us)
    N = num_agents(dyn)
    @assert N == length(us)
    @assert size(x, 1) == 4 * N
    for ii in 1:N
        @assert size(us[ii], 1) == 2
    end
    num_states_per_player = Int(xdim(dyn) / N)
    As = [sparse(zeros(num_states_per_player, num_states_per_player)) for ii in 1:N]
    for ii in 1:N
        start_idx = num_states_per_player * (ii-1)
        theta = x[start_idx + 3]
        v = x[start_idx + 4]

        # Compute the state and controls for each actor.
        s = sin(theta)
        c = cos(theta)
        As[ii][1:2, 3:4] = [-v*s c; v*c s]
    end
    return Matrix(blockdiag(As...))
end

function Fus(dyn::UnicycleDynamics, time_range, x, us)
    N = num_agents(dyn)
    @assert N == length(us)
    @assert size(x, 1) == 4 * N
    for ii in 1:N
        @assert size(us[ii], 1) == 2
    end
    num_states_per_player = Int(xdim(dyn) / N)
    Bs = [zeros(xdim(dyn), udim(dyn, ii)) for ii in 1:N]
    prev_time, curr_time = time_range
    dt = curr_time - prev_time
    for ii in 1:N
        start_idx = num_states_per_player * (ii-1)
        Bs[ii][start_idx+3:start_idx+4, 1:2] = [1 0; 0 1]
    end
    return Bs
end

export dx, propagate_dynamics, Fx, Fus

# TODO(hamzah) - refactor this to adjust based on number of players instead of assuming 2.
function plot_states_and_controls(dyn::UnicycleDynamics, times, xs, us)
    @assert num_agents(dyn) == 2

    x₁ = xs[:, 1]

    title1 = "pos. traj."
    q1 = plot(legend=:outertopright, title=title1, xlabel="x (m)", ylabel="y (m)")
    plot!(q1, xs[1, :], xs[2, :], label="P1 pos")
    plot!(q1, xs[5, :], xs[6, :], label="P2 pos")

    q1 = scatter!([x₁[1]], [x₁[2]], color="blue", label="start P1")
    q1 = scatter!([x₁[5]], [x₁[6]], color="red", label="start P2")

    title2a = "x-pos"
    q2a = plot(legend=:outertopright, title=title2a, xlabel="t (s)", ylabel="x (m)")
    plot!(times, xs[1,:], label="P1 px")
    plot!(times, xs[5,:], label="P2 px")

    title2b = "y-pos"
    q2b = plot(legend=:outertopright, title=title2b, xlabel="t (s)", ylabel="y (m)")
    plot!(times, xs[2,:], label="P1 py")
    plot!(times, xs[6,:], label="P2 py")

    title3 = "θ"
    q3 = plot(legend=:outertopright, title=title3, xlabel="t (s)", ylabel="θ (rad)")
    plot!(times, xs[3,:], label="P1 θ")
    plot!(times, xs[7,:], label="P2 θ")

    title4 = "vel"
    q4 = plot(legend=:outertopright, title=title4, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[4,:], label="P1 v")
    plot!(times, xs[8,:], label="P2 v")

    title5 = "ang vel"
    q5 = plot(legend=:outertopright, title=title5, xlabel="t (s)", ylabel="ang. vel. (rad/s)")
    plot!(times, us[1][1, :], label="P1 ω")
    plot!(times, us[2][1, :], label="P2 ω")

    title6 = "accel"
    q6 = plot(legend=:outertopright, title=title6, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][2, :], label="P1 accel")
    plot!(times, us[2][2, :], label="P2 accel")

    return q1, q2a, q2b, q3, q4, q5, q6
end

export plot_states_and_controls
