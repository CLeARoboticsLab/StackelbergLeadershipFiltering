# Utilities

# TODO: Add a better way to integrate a homogenized matrix into the rest of the system
# Produces a symmetric matrix.
# If we need to perform a spectral shift to enforce PD-ness, we can set rho accordingly.
function homogenize_matrix(M::AbstractMatrix{Float64}, m::AbstractVector{Float64}, cm::Float64; ρ=0.0)
    M_dim = size(M, 1)
    return vcat(hcat(M ,  m),
                hcat(m', cm)) + ρ * I
end

export homogenize_matrix


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

function udim(sys_info::SystemInfo, player_idx)
    return sys_info.num_us[player_idx]
end

function vdim(sys_info::SystemInfo)
    return sys_info.num_v
end

export SystemInfo, num_agents, xdim, udim, vdim
