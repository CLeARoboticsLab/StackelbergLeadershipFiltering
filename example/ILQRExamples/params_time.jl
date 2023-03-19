num_players = 1

T = 101
t0 = 0.0
dt = 0.05
horizon = T * dt
times = dt * cumsum(ones(T)) .- dt
