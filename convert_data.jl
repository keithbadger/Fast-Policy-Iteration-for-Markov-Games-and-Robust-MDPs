using Plots, CSV, DataFrames, Arrow

filenames = ["mg_all.arrow", "inv_all.arrow", "ruin_all.arrow", "grid_all.arrow","games_large.arrow", "inventory_large.arrow", "ruin_large.arrow", "gridworld_large.arrow"]
for filename ∈ filenames
    results = copy(DataFrame(Arrow.Table("data/"*filename)))
    replace!(results.algorithm, "RCPI" => "RCPI₀", "KB" => "RCPI∞", "WIN" => "WS")
    Arrow.write("data2/"*filename,results)
end