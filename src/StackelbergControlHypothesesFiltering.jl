module StackelbergControlHypothesesFiltering

include("Utils.jl")

include("ilqr.jl")
include("StackelbergILQGames.jl")

include("costs/CostUtils.jl")
include("costs/QuadraticCost.jl")
include("costs/AffineCost.jl")

include("dynamics/DynamicsUtils.jl")
include("dynamics/LinearDynamics.jl")
include("dynamics/UnicycleDynamics.jl")

include("control_strategies/ControlStrategyUtils.jl")
include("control_strategies/FeedbackGainControlStrategy.jl")

include("solvers/LQNashFeedbackSolver.jl")
include("solvers/LQStackelbergFeedbackSolver.jl")
include("solvers/LQRFeedbackSolver.jl")

end # module
