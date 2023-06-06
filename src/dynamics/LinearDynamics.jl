# TODO(hamzah) Add better tests for the LinearDynamics struct and associated functions.
# Describes the dynamics x_{t+1} = Ax_t + a + ∑Buᵢ_t + Dv_t; x = state, u = inputs, v = process noise
struct LinearDynamics <: Dynamics
    A  # linear state dynamics term
    a  # constant state dynamics term
    Bs # controls
    D  # process noise
    sys_info::SystemInfo
end

# TODO(hamzah) Add [:,:] as necessary for auto-sizing - fixes bug if 1D vector passed in when a 2D matrix is expected.
# Constructor for linear dynamics that auto-generates the system info and has no process noise.
# TODO(hamzah): Add an optional bs argument that defaults to zero vector of proper size.
ContinuousLinearDynamics(A, Bs; a=zeros(size(A, 1))) = LinearDynamics(A, a, Bs, nothing,
                                                SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)]))

# Constructor for linear dynamics that is provided the system info and has no process noise.
LinearDynamics(A, Bs, sys_info::SystemInfo; a=zeros(size(A, 1))) = LinearDynamics(A, a, Bs, nothing, sys_info)

# Constructor for linear dynamics that auto-generates the system info with process noise.
ContinuousLinearDynamics(A, Bs, D; a=zeros(size(A, 1))) = LinearDynamics(A, a, Bs, D,
                                                SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)], size(D, 2), 0.0))

export ContinuousLinearDynamics, LinearDynamics

function dx(dyn::LinearDynamics, time_range, x, us, v)
    @assert is_continuous(dyn) "dx is defined on continuous-time dynamics objects."
    dfdx = dyn.A * x + dyn.a + sum(get_control_dynamics(dyn, ii) * us[ii] for ii in 1:N)
    if dyn.D != nothing && v != nothing
        dfdx += dyn.D * v
    end
    return dfdx
end

function propagate_dynamics(dyn::LinearDynamics, time_range, x, us, v)
    @assert !is_continuous(dyn) "Only discrete-time dynamics objects can be propagated."
    N = num_agents(dyn)

    # Assertions to confirm sizes.
    @assert size(x, 1) == xdim(dyn)
    for ii in 1:N
        @assert size(us[ii], 1) == udim(dyn, ii)
    end

    # Incorporate the dynamics based on the state and the controls.
    x_next = dyn.A * x + dyn.a + sum(dyn.Bs[ii] * us[ii] for ii in 1:N)

    if dyn.D != nothing && v != nothing
        x_next += dyn.D * v
    end

    # Remove the extra dimension before returning the propagated state.
    return x_next
end

# Produces a continuous-time Jacobian linearized system from any linear system.
function linearize(dyn::LinearDynamics, time_range, x, us)
    return dyn
end

function discretize(dyn::LinearDynamics, dt::Float64)
    # If is already discretized at the correct sample rate, then return the system as-is.
    if !is_continuous(dyn) && dt == sampling_time(dyn)
        return dyn
    end

    @assert is_continuous(dyn) "Input dynamics must be continuous to be discretized."
    new_sys_info = get_discretized_system_info(dyn.sys_info, dt)
    new_D = isnothing(dyn.D) ? nothing : dt * dyn.D
    return LinearDynamics(I + dt * dyn.A, dt * dyn.a, dt * dyn.Bs, new_D, new_sys_info)
end

# A function which jointly linearizes and discretizes any dynamics.
function linearize_discretize(dyn::LinearDynamics, time_range, x, us)
    return discretize(dyn, new_sampling_time)
end

# TODO(hmzh) - may need dfdv as well


export propagate_dynamics, linearize, discretize, linearize_discretize

using Plots
# TODO(hamzah) - refactor this to be tied DoubleIntegrator Dynamics instead of Linear Dynamics.
function plot_states_and_controls(dyn::LinearDynamics, times, xs, us)
    @assert num_agents(dyn) == 2
    @assert xdim(dyn) == 8
    @assert udim(dyn, 1) == 2
    @assert udim(dyn, 2) == 2
    x₁ = xs[:, 1]

    title1 = "pos. traj."
    q1 = plot(legend=:outertopright, title=title1, xlabel="x (m)", ylabel="y (m)")
    plot!(q1, xs[1, :], xs[3, :], label="P1 pos")
    plot!(q1, xs[5, :], xs[7, :], label="P2 pos")

    q1 = scatter!([x₁[1]], [x₁[3]], color="blue", label="start P1")
    q1 = scatter!([x₁[5]], [x₁[7]], color="red", label="start P2")

    title2a = "x-pos"
    q2a = plot(legend=:outertopright, title=title2a, xlabel="t (s)", ylabel="x (m)")
    plot!(times, xs[1,:], label="P1 px")
    plot!(times, xs[5,:], label="P2 px")

    title2b = "y-pos"
    q2b = plot(legend=:outertopright, title=title2b, xlabel="t (s)", ylabel="y (m)")
    plot!(times, xs[3,:], label="P1 py")
    plot!(times, xs[7,:], label="P2 py")

    title3 = "x-vel"
    q3 = plot(legend=:outertopright, title=title3, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[2,:], label="P1 vx")
    plot!(times, xs[6,:], label="P2 vx")

    title4 = "y-vel"
    q4 = plot(legend=:outertopright, title=title4, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[4,:], label="P1 vy")
    plot!(times, xs[8,:], label="P2 vy")

    title5 = "x-accel"
    q5 = plot(legend=:outertopright, title=title5, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][1, :], label="P1 ax")
    plot!(times, us[2][1, :], label="P2 ax")

    title6 = "y-accel"
    q6 = plot(legend=:outertopright, title=title6, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][2, :], label="P1 ay")
    plot!(times, us[2][2, :], label="P2 ay")

    return q1, q2a, q2b, q3, q4, q5, q6
end

export plot_states_and_controls

