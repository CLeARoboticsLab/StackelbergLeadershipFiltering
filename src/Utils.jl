# Utilities


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

function vdim(sys_info::SystemInfo)
    return sys_info.num_v
end

function sampling_time(sys_info::SystemInfo)
    return sys_info.dt
end

function is_continuous(sys_info::SystemInfo)
    return iszero(sampling_time(sys_info))
end

function get_discretized_system_info(dyn::Dynamics, new_dt)
    @assert is_continuous(dyn) "Input dynamics must be continuous to be discretized."
    return SystemInfo(dyn.sys_info, new_dt)
end

export SystemInfo, num_agents, xdim, udim, vdim, sampling_time, is_continuous, get_discretized_system_info


# Wraps angles to the range [-pi, pi).
function wrap_angle(angle_rad)
    wrapped_angle = (angle_rad + pi) % (2*pi)
    return wrapped_angle - pi
end

export wrap_angle
