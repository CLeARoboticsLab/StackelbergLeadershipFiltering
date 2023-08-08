# Utilities

using Dates

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
    while ang >= 2*pi
        ang -= 2*pi
    end
    while ang < 0
        ang += 2*pi
    end
    return ang - pi

    # ang = angle_rad
    # while abs(angle_rad) >= pi
    #     ang -= sign(angle_rad) * 2 * pi
    # end
    return ang
end

export wrap_angle



function get_date_str()
    # Get the current date and time
    current_datetime = Dates.now()

    # Extract the month, day, hour, and minute
    current_month = Dates.month(current_datetime)
    current_day = Dates.day(current_datetime)
    current_hour = Dates.hour(current_datetime)
    current_minute = Dates.minute(current_datetime)

    # Create a formatted string
    return string(current_month, "_", current_day, "_", current_hour, "_", current_minute)
end
export get_date_str
