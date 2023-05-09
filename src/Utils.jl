# Utilities

# A function that splits 2D arrays into vectors of 2D arrays
# Use case. combining and splitting control inputs.
function split(x, n)
    @assert sum(n) == size(x, 1)

    result = Vector{Matrix{eltype(x)}}()
    start = firstindex(x)
    for len in n
        push!(result, x[start:(start + len - 1), :])
        start += len
    end
    return result
end


# TODO(hmzh) Add a game class of some sort that ties together the system info, cost, and dynamics, factoring in possible
#            homogenization.
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

function num_us(sys_info::SystemInfo)
    return sys_info.num_us
end

function udim(sys_info::SystemInfo, player_idx)
    return sys_info.num_us[player_idx]
end

function vdim(sys_info::SystemInfo)
    return sys_info.num_v
end

# For homogenized dimensions
function xhdim(sys_info::SystemInfo)
    return sys_info.num_x + 1
end

function uhdim(sys_info::SystemInfo)
    return sum(uhdim(sys_info, ii) for ii in 1:num_agents(sys_info))
end

function uhdim(sys_info::SystemInfo, player_idx)
    return sys_info.num_us[player_idx] + 1
end

export SystemInfo, num_agents, xdim, udim, num_us, vdim, xhdim, uhdim


function homogenize_vector(v::AbstractVector{Float64})
    return vcat(v, 1)
end

function homogenize_vector(vs::AbstractMatrix{Float64})
    return vcat(vs, ones(1, size(vs, 2)))
end

export homogenize_vector


# Wraps angles to the range [-pi, pi).
function wrap_angle(angle_rad)
    wrapped_angle = (angle_rad + pi) % (2*pi)
    return wrapped_angle - pi
end

export wrap_angle
