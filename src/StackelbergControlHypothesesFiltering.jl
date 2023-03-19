module StackelbergControlHypothesesFiltering

include("Utils.jl")

include("costs/CostUtils.jl")
include("costs/QuadraticCost.jl")
include("costs/QuadraticCostWithOffset.jl")

include("costs/examples/ExampleILQRCost.jl")

include("dynamics/DynamicsUtils.jl")
include("dynamics/LinearDynamics.jl")
include("dynamics/UnicycleDynamics.jl")

include("control_strategies/ControlStrategyUtils.jl")
include("control_strategies/FeedbackGainControlStrategy.jl")

include("games/ShepherdAndSheepGame.jl")

include("solvers/IterativeLQR.jl")
include("solvers/LQNashFeedbackSolver.jl")
include("solvers/LQStackelbergFeedbackSolver.jl")
include("solvers/LQRFeedbackSolver.jl")
include("solvers/StackelbergIterativeLQGames.jl")

include("TwoStateParticleFilter.jl")

end # module
