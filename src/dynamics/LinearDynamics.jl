# TODO(hamzah) Add better tests for the LinearDynamics struct and associated functions.
struct LinearDynamics <: Dynamics
    A  # state
    Bs # controls
    D  # process noise
    sys_info::SystemInfo
end

# TODO(hamzah) Add [:,:] as necessary for auto-sizing - fixes bug if 1D vector passed in when a 2D matrix is expected.
# Constructor for linear dynamics that auto-generates the system info and has no process noise.
LinearDynamics(A, Bs) = LinearDynamics(A, Bs, nothing,
                                       SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)]))

# Constructor for linear dynamics that is provided the system info and has no process noise.
LinearDynamics(A, Bs, sys_info::SystemInfo) = LinearDynamics(A, Bs, nothing, sys_info)

# Constructor for linear dynamics that auto-generates the system info with process noise.
LinearDynamics(A, Bs, D) = LinearDynamics(A, Bs, D,
                                          SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)], size(D, 2)))

# A function definition that does not accept process noise input and reroutes to the type-specific propagate_dynamics that does.
function propagate_dynamics(dyn::Dynamics,
                            time_range,
                            x::AbstractVector{Float64},
                            us::AbstractVector{<:AbstractVector{Float64}})
    # Ensure that there should not be any process noise.
    @assert vdim(dyn) == 0
    @assert time_range[1] ≤ time_range[2]

    return propagate_dynamics(dyn, time_range, x, us, nothing)
end

function propagate_dynamics(dyn::LinearDynamics,
                            time_range,
                            x::AbstractVector{Float64},
                            us::AbstractVector{<:AbstractVector{Float64}},
                            v::Union{Nothing, AbstractVector{Float64}})
    N = dyn.sys_info.num_agents
    x_next = dyn.A * x
    for i in 1:N
        ui = reshape(us[i], dyn.sys_info.num_us[i], 1)
        x_next += dyn.Bs[i] * ui
    end

    if dyn.D != nothing && v != nothing
        x_next += dyn.D * v
    end

    return x_next
end

function linearize_dynamics(dyn::LinearDynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return dyn
end

export LinearDynamics, propagate_dynamics, linearize_dynamics
