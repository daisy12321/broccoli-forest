using JuMP, Gurobi
using Distributions

srand(4)

function solve_problem1(A, b, m=20,n=30,k=10)
	####### Problem 1a: 		#############
	# solve original l0 problem

	print("Building optimization model\n")

	m = Model(solver = GurobiSolver());

	@defVar(m, x[1:n]);
	@defVar(m, z[1:n], Bin);

	# Big M constraints; if z = 1, then x = 0
	for i=1:n
		@addConstraint(m, x[i]  <= BigM*(1-z[i]));
		@addConstraint(m, -x[i] <= BigM*(1-z[i]));
	end

	# original constraint
	@addConstraint(m, A*x .== b);

	# maximize the number of zero elements
	@setObjective(m, :Max, sum{z[i],i=1:n});

	print("\NDone building optimization model\n")
	status = solve(m)

	x_sol = getValue(x[:])
	countnz(x_sol)
	return(x_sol)
end

function solve_problem2(A, b, m=20,n=30,k=10)
	####### Problem 1b: 		#############
	# l1 relaxation

	print("Building optimization model\n")

	m2 = Model(solver = GurobiSolver());

	@defVar(m2, xpos[1:n] >= 0);
	@defVar(m2, xneg[1:n] >= 0);

	# original constraint
	@addConstraint(m2, A*(xpos-xneg) .== b);

	# maximize the number of zero elements
	@setObjective(m2, :Min, sum{xpos[k]+xneg[k],k=1:n});

	print("\NDone building optimization model\n")
	status = solve(m2)

	x_sol2 = getValue(xpos[:]-xneg[:])
	countnz(x_sol2)
	# correctly reocover!
	return(x_sol2)
end

n = 100
# m = 10
# k = 5
# x_sol, x_sol2 =  solve_problem1(m,n,k)
d = Normal(0, 1)		

sol_recovered = fill(0, convert(Int, n/10), 10)
sol_header = [1:10]' #k/m
sol_rowname = [0:(n/10)]
for m=10:10:n
	for k_idx=1:10
		k = convert(Int, m/10*k_idx)

		A = rand(d, m, n)
		x0 = rand(d, n)
		# k non-zero elements
		x0[sample(1:n,n-k,replace=false)] = 0
		x0
		countnz(x0)

		b = A*x0
		BigM = 100

		# x_sol =  solve_problem1(A, b, m,n,k)
		x_sol2 = solve_problem2(A, b, m,n,k)
		# if countnz(x_sol) == countnz(x_sol2)
		if countnz(x_sol2) == k
			sol_recovered[convert(Int, m/10), k_idx] = true
		end
	end
end
output = [sol_header; sol_recovered]
output_full = [sol_rowname/10 output]
writecsv("hw1_output.csv", output_full)
