# Utilities


# TODO(hmzh) Add a game class of some sort that ties together the system info, cost, and dynamics.
struct SystemInfo
    num_agents::Int
    num_x::Int
    num_us::AbstractArray{Int}
    num_v::Int
end
SystemInfo(num_agents, num_x, num_us) = SystemInfo(num_agents, num_x, num_us, 0)

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

export SystemInfo, num_agents, xdim, udim, vdim


# Wraps angles to the range [-pi, pi).
function wrap_angle(angle_rad)
    wrapped_angle = (angle_rad + pi) % (2*pi)
    return wrapped_angle - pi
end

export wrap_angle
