using JuMP, HiGHS, Plots

# From Table 1, Workload requirements for numerical results
# W[j, k] = workload for shift j on day k
W = [
    19 16 22 22 22 22 22; # Morning (M)
    19 16 22 22 22 22 22; # Evening (E)
    14 11 16 16 16 16 16  # Night (N)
]

# A[j, k] = additional staff needed for shift j on day k
A = fill(100, 3, 7)

days = 1:7
shifts = 1:3


function solve_for_n_employees(N)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # x[i,j,k]
    @variable(model, x[1:N, shifts, days], Bin)

    # Minimize total number of assigned shifts
    @objective(model, Min, sum(x))

    # Constraints
    # Eq. (8) - (10): Num consecutive shifts each employee ≤ 2
    max_consecutive_shifts = 2

    @constraint(model, [i = 1:N, k = days], sum(x[i, :, k]) <= max_consecutive_shifts)

    @constraint(
        model,
        [i = 1:N, k = days],
        x[i, 2, k] + x[i, 3, k] + x[i, 1, mod(k, 7)+1] <= max_consecutive_shifts
    )

    @constraint(
        model,
        [i = 1:N, k = days],
        x[i, 3, k] + x[i, 1, mod(k, 7)+1] + x[i, 2, mod(k, 7)+1] <= max_consecutive_shifts
    )

    # Eq. (11): Num shifts worked by each employee ≥ workload
    @constraint(model, [j = shifts, k = days], sum(x[:, j, k]) >= W[j, k])

    # Eq. (12): Num shifts worked by each employee ≤ workload + additional staff
    @constraint(model, [j = shifts, k = days], sum(x[:, j, k]) <= W[j, k] + A[j, k])


    # Eq. (13) Each employee should have 2 consecutive days off
    @constraint(model, [i = 1:N],
        sum(
            (x[i, j, mod(i - 1, 7)+1] + x[i, j, mod(i, 7)+1])
            for j in 1:3
        ) == 0
    )

    # Eq. (14) Each employee should work at least 1 shift per day
    @constraint(model, [i = 1:N, k = 1:5],
        sum(x[i, j, mod(i + k, 7)+1] for j in shifts) >= 1
    )

    optimize!(model)

    return model
end


# Recreating results from Hoseini et al. for N = 70:90
N_range = 70:90
M = 395     # Total workload
rO = 0.245  # Cost factor for overtime

computed_costs = []
lower_bound_costs = []

lower_bound(N) = N + (max(0, (M - 5 * N)) * rO)

for N in N_range
    model = solve_for_n_employees(N)
    total_shifts = objective_value(model)
    overtime_shifts = max(0, total_shifts - (5 * N))
    cost = N + (overtime_shifts * rO)

    push!(computed_costs, cost)
    push!(lower_bound_costs, lower_bound(N))
end

p = plot(
    N_range, lower_bound_costs,
    label="Lower Bound",
    color=:black,
    linewidth=1.0,
    linestyle=:solid,
    legend=:topleft,
    framestyle=:box,
    grid=false,
    xticks=70:2:90,
    ylim=(78, 91),
    xlabel="Number of Employees (N)",
    ylabel="Normalized Total Cost (C/S)",
    fontfamily="Computer Modern",
    guidefontsize=10,
    tickfontsize=9,
    legendfontsize=9,
    size=(600, 450)
)

plot!(
    N_range, computed_costs,
    label="Computed Solution",
    seriestype=:line,
    color=:navy,
    linewidth=1.0,
    markershape=:circle,
    markercolor=:navy,
    markersize=4,
    markerstrokewidth=0,
    grid=true,
    gridalpha=0.1,
    gridstyle=:dash,
)

# Highlight the optimal solution 
min_cost, idx = findmin(computed_costs)
optimal_N = N_range[idx]

scatter!(
    [optimal_N], [min_cost],
    color=:red,
    markersize=3,
    markerstrokewidth=0,
    label="Optimal Solution (N=$optimal_N)"
)

display(p)