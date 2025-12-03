include("rmdp_algorithm_comparison.jl")

using DataFrames, Arrow

algorithms = [(name = "PAI",), (name = "RCPI∞",), (name = "RCPI₀",), (name = "FT", η = 1e-3, β = .5), (name = "VI",)] 
output_inventory = benchmark_run_inventory(algorithms, repeat(40:40:200, inner = 2), [.9 ,.99, .999],1,1e-3,28800.)
Arrow.write("data/inventory_large.arrow", output_inventory)
output_ruin = benchmark_run_ruin(algorithms, repeat(20:20:100, inner = 2), [.9 ,.99, .999],1,1e-3,28800.)
Arrow.write("data/ruin_large.arrow", output_ruin)
output_grid = benchmark_run_gridworld(algorithms, repeat(4:4:20, inner = 2), [.9 ,.99, .999],1,1e-3,28800.)
Arrow.write("data/gridworld_large.arrow", output_grid)

algorithms = [(name = "PAI",), (name = "RCPI∞",), (name = "RCPI₀",), (name = "FT", η = 1e-3, β = .5), (name = "VI",), (name = "WS", H = 10, m = 100), (name = "HK",), (name = "PPI", ϵ₂ = .1, β = .5)]
output_inventory = benchmark_run_inventory(algorithms, repeat(4:4:20, inner = 2), [.5, .75, .9 ,.99],1,1e-3,28800.)
Arrow.write("data/inventory_all.arrow", output_inventory)
output_ruin = benchmark_run_ruin(algorithms, repeat(10:10:50, inner = 2), [.5, .75, .9 ,.99],1,1e-3,28800.)
Arrow.write("data/ruin_all.arrow", output_ruin)
output_grid = benchmark_run_gridworld(algorithms, repeat(2:2:10, inner = 2), [.5, .75, .9 ,.99],1,1e-3,28800.)
Arrow.write("data/gridworld_all.arrow", output_grid)