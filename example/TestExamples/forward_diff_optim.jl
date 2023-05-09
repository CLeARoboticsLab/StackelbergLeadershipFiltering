using DifferentialEquations
using ForwardDiff

function CostFun(x::AbstractVector{T}) where T
           
    function SpringEqu!(du, u, x, t)
        du[1] = u[2]
        du[2] = -(x[1] / x[3]) * u[2] - (x[2] / x[3]) * u[1] + 50 / x[3]
    end

    u0 = T[2.0, 0.0]
    tspan = (0.0, 1.0)
    prob = ODEProblem(SpringEqu!, u0, tspan, x)
    sol = solve(prob)


    Simpos = zeros(T, length(sol.t))
    Simvel = zeros(T, length(sol.t))
    tout = zeros(T, length(sol.t))
    for i = 1:length(sol.t)
        tout[i] = sol.t[i]
        Simpos[i] = sol[1, i]
        Simvel[i] = sol[2, i]
    end

    totalCost = sum(Simpos)
    return totalCost
end

xin = [2000.0, 20000.0, 80.0]
g = ForwardDiff.gradient(CostFun, xin)
