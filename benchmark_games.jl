include("game_algorithm_comparison.jl")


using DataFrames, Arrow


algorithms = [(name = "PAI",), (name = "RCPI∞",), (name = "RCPI₀",), (name = "FT", η = 1e-3, β = .5), (name = "VI",)] 
output_games = benchmark_run_games(repeat(200:200:1000, inner = 2), [.9 ,.99, #= .999 =#], algorithms, [1,2,3,5,10], -10., 10., .2, 1e-3, 28800.)
Arrow.write("data2/games_large.arrow", output_games)

algorithms = [(name = "PAI",), (name = "RCPI∞",), (name = "RCPI₀",), (name = "FT", η = 1e-3, β = .5), (name = "VI",), (name = "WS", H = 10, m = 100), (name = "HK",), (name = "PPI", ϵ₂ = .1, β = .5)] 
output_games = benchmark_run_games(repeat(20:20:100, inner = 2), [.5, .75, .9 ,.99], algorithms, [1,2,3,5,10], -10., 10., .2, 1e-3, 28800.)
Arrow.write("data2/games_all.arrow", output_games)
