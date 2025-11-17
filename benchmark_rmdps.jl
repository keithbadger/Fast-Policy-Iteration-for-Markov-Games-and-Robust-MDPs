include("rmdp_algorithm_comparison.jl")

using DataFrames, Arrow

algorithms = [(name = "PAI",), (name = "KB",), (name = "RCPI",), (name = "FT", η = 1e-3, β = .5), (name = "VI",)]
output_inventory = benchmark_run_inventory(algorithms, repeat(40:40:200, inner = 20), [.9 ,.99, .999],1,1e-3,28800.)
Arrow.write("data/inventory_large.arrow", output_inventory)
output_ruin = benchmark_run_ruin(algorithms, repeat(20:20:100, inner = 20), [.9 ,.99, .999],1,1e-3,28800.)
Arrow.write("data/ruin_large.arrow", output_ruin)
output_grid = benchmark_run_gridworld(algorithms, repeat(4:4:20, inner = 20), [.9 ,.99, .999],1,1e-3,28800.)
Arrow.write("data/gridworld_large.arrow", output_grid)
