# Utilities

# TODO(hamzah) - Combine split_vec and split functions.

# A function that splits 2D arrays into vectors of 2D arrays
# Use case. combining and splitting control inputs.
function split(x, n)
    if ndims(x) == 1
        return split_vec(x, n)
    end

    @assert sum(n) == size(x, 1)

    result = Vector{Matrix{eltype(x)}}()
    start = firstindex(x)
    for len in n
        push!(result, x[start:(start + len - 1), :])
        start += len
    end
    return result
end

function split_vec(x::AbstractVector{T}, n) where {T}
    @assert sum(n) == size(x, 1)

    result = Vector{Vector{eltype(x)}}()
    start = firstindex(x)
    for len in n
        push!(result, x[start:(start + len - 1)])
        start += len
    end
    return result
end

export split

# TODO(hmzh) Add a game class of some sort that ties together the system info, cost, and dynamics.
struct SystemInfo
    num_agents::Int
    num_x::Int
    num_us::AbstractArray{Int}
    num_v::Int
    dt::Float64 # If this is set to 0, the system is continuous.
end
SystemInfo(num_agents, num_x, num_us, dt=0.) = SystemInfo(num_agents, num_x, num_us, 0, dt)
SystemInfo(si::SystemInfo, dt) = SystemInfo(si.num_agents, si.num_x, si.num_us, si.num_v, dt)

function num_agents(sys_info::SystemInfo)
    return sys_info.num_agents
end

function xdim(sys_info::SystemInfo)
    return sys_info.num_x
end

function udim(sys_info::SystemInfo)
    return sum(sys_info.num_us)
end

function udim(sys_info::SystemInfo, player_idx)
    return sys_info.num_us[player_idx]
end

function udims(sys_info::SystemInfo)
    return sys_info.num_us
end

function vdim(sys_info::SystemInfo)
    return sys_info.num_v
end

function sampling_time(sys_info::SystemInfo)
    return sys_info.dt
end

function is_continuous(sys_info::SystemInfo)
    return iszero(sampling_time(sys_info))
end

function get_discretized_system_info(sys_info::SystemInfo, new_dt)
    return SystemInfo(sys_info, new_dt)
end

export SystemInfo, num_agents, xdim, udim, udims, vdim, sampling_time, is_continuous, get_discretized_system_info


# Wraps angles to the range [-pi, pi).
# Note: implementations using the mod operator cause havoc with autodiff.
function wrap_angle(angle_rad)
    ang = angle_rad + pi
    while angle_rad >= 2*pi
        angle_rad -= 2*pi
    end
    while angle_rad < 0
        angle_rad += 2*pi
    end
    return angle_rad - pi

    # ang = angle_rad
    # while abs(angle_rad) >= pi
    #     ang -= sign(angle_rad) * 2 * pi
    # end
    return ang
end

export wrap_angle
