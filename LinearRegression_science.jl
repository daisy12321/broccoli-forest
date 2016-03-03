using JuMP, Gurobi
using GLM
using DataFrames
include("FirstOrderHeuristic.jl")
tic()
####################################
# CHANGE THE FOLLOWING FOR YOUR PARTICULAR DATA:

# BIG M CONSTANT
M = 50
LAMBDA = 0.1;

include("LinearRegression_read_data.jl")
########## Finish data cleaning ################################

# data_train = DataFrame([y_train X_train_new])
# glm(:x1 ~ :x2, data = data_train)
#######


#### NEW: group multi-colinearity
group_1 = [1:D]
group_2 = [D+1:2D]
group_3 = [2D+1:3D]
###### NEW: preprocess the data:
#### correlation to identify pairwise multicolinear 
CORR_THRESHOLD = 0.8
X_train_cor = cor(X_train_new)
X_train_cor_ut = triu(X_train_cor) - eye(3*D)
pairs_1, pairs_2 = ind2sub(size(X_train_cor_ut), find(abs(X_train_cor_ut) .>= CORR_THRESHOLD))

#######
# Build MIO optimization model

print("Building optimization model\n")

m = Model(solver = GurobiSolver(OutputFlag =0 ))

# to specify any solver parameters, add them like this:
# m = Model(solver = GurobiSolver(TimeLimit = 10, MIPGap = 0.01))

@defVar(m, Beta[1:3*D]);
@defVar(m, z[1:3*D], Bin);
@defVar(m, BetaAbs[1:3*D]);

# Big M constraints
for d=1:3*D
	@addConstraint(m, Beta[d] <= M*z[d]);
	@addConstraint(m, -M*z[d] <= Beta[d]);
end
@addConstraint(m, abs_cstr1[d=1:3D], BetaAbs[d]>= Beta[d]);
@addConstraint(m, abs_cstr2[d=1:3D], BetaAbs[d]>= -Beta[d]);


### NEW: to add the pairwise colinearity constraint
for j=1:length(pairs_1)
	@addConstraint(m, z[pairs_1[j]]+z[pairs_2[j]] <= 1);
end

### NEW: to add the group colinearity constraint on non-linear transformations
for j=1:length(group_1)
	@addConstraint(m, z[group_1[j]]+z[group_2[j]]+z[group_3[j]] <= 1);
end

# Sparsity constraint
@addConstraint(m, sparsity, sum{z[d], d=1:3*D} <= K_options[1])

# Objective function

a = 0	
for i=1:N
	a += 0.5(y_train[i] - dot(Beta, vec(X_train_new[i,:])))^2 
end


### NEW: add robust term; for now assume lambda = 10
setObjective(m, :Min, a+LAMBDA*sum(BetaAbs))

print("\NDone building optimization model\n")

Betavals = zeros(3*D,3*D)
MIO_num_real_RSS = 0
bestR2 = 0
R2_test = 0
bestBetavals = zeros(3*D)

for K in K_options
	print("\nstarting to solve k = ", K)
	chgConstrRHS(sparsity, K) #Sparsity constraint


	print("\n getting warm start solution\n")
	betaWarm = WarmStart(X_train_new, y_train, K)
	zWarm = 1*(betaWarm .!= 0)
	for d=1:3*D
		setValue(Beta[d], betaWarm[d])
		setValue(z[d], zWarm[d])
	end
	print("\n set warm start solution\n")

	num_no_zero = 3*D + 1
	t_val = zeros(3*D)

	while (sum(abs(t_val) .<= 1.98) > 0)
		print("\n Looping\n")

		# add constraint 
		@addConstraint(m, sum{z[d], d=1:3*D} <= num_no_zero - 1)

		# solve the problem
		status = solve(m)
		beta_tmp = getValue(Beta)

		# find the t-stats
		idx_no_zero = find(beta_tmp .!= 0)
		num_no_zero = countnz(idx_no_zero)

		X_train_sub = X_train_new[:,idx_no_zero]
		betaHat = inv(X_train_sub'*X_train_sub)*X_train_sub'*y_train
		y_train_pred = X_train_sub *betaHat
		SSE = dot(y_train-y_train_pred, y_train-y_train_pred)
		var_betaHat = diag(SSE/(N-num_no_zero-1) *inv(X_train_sub'*X_train_sub))
		t_val = betaHat ./sqrt(var_betaHat)
		println("t test: ",t_val)

	end

	for j=1:3*D
		Betavals[K, j] = getValue(Beta[j])
	end 

	y_hat_validation = X_validation_new*Betavals[K,:]'
	RSS_current = sum((y_hat_validation - y_validation).^2)
	SST = sum((y_validation - mean(y_train)).^2)

	newR2 = 1-RSS_current/SST
					
	if (newR2 > bestR2)					
		bestR2 = newR2
		bestBetavals = Betavals[K,:]
	end		

end


# Out of sample testing
y_hat_test = X_test_new*bestBetavals'
RSS_test = sum((y_hat_test - y_test).^2)
R2_test = 1- RSS_test/SST_test
MIO_nonzeros = find(abs(bestBetavals) .> 0.00001)

print("\n\n***RESULTS***")
print("\n N:\t", N)
print("\n D:\t", D)
print("\nbest K\t", length(MIO_nonzeros))
print("\nMIO R2 test\t", R2_test)
print("\n")	
toc()