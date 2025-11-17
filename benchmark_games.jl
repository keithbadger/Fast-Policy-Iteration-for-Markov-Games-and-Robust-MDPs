include("game_algorithm_comparison.jl")


using DataFrames, Arrow


algorithms = [(name = "PAI",), (name = "KB",), (name = "RCPI",), (name = "FT", η = 1e-3, β = .5), (name = "VI",)] 
output_games = benchmark_run_games(repeat(200:200:1000, inner = 20), [.9 ,.99, .999], algorithms, [1,2,3,5,10], -10., 10., .2, 1e-3, 28800.)
Arrow.write("data/games_large.arrow", output_games)

