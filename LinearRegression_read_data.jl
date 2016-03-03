
# PATH TO YOUR DATASETS HERE
trainfile = "Train lpga2009_opt.csv" 
validationfile = "Validation lpga2009_opt.csv" 
testfile = "Test lpga2009_opt.csv"

####################################

train = readcsv(trainfile)
validation = readcsv(validationfile)
test = readcsv(testfile)
print("\nDone reading in data\n")

D = size(train)[2] - 1	
N = size(train)[1]
y_train = train[:, 1]
X_train = train[:,2:(D+1)]
y_validation = validation[:, 1]
X_validation = validation[:,2:(D+1)]
y_test = test[:,1]
X_test = test[:, 2:(D+1)]


# determine centering and scaling factors
mean_X_train = mean(X_train,1);
mean_y_train = mean(y_train);
X_train = X_train .- mean_X_train;
denom_X_train = zeros(D);
for i in 1:D
	denom_X_train[i] = norm(X_train[:,i]);
end

# center and scale the datasets
X_train = X_train ./ denom_X_train';
X_validation = (X_validation .- mean_X_train)./denom_X_train'
X_test = (X_test .- mean_X_train)./denom_X_train'
y_train = y_train .- mean_y_train;
y_validation = y_validation .- mean_y_train
y_test = y_test .- mean_y_train

SST_test = sum((mean(y_train) - y_test).^2)
K_options = reverse([1:D])


###### NEW: add non-linear transformations
X_train_new = zeros(N, D*3)
X_train_new[1:N, 1:D] = copy(X_train)
for d=1:D
	X_train_new[:,D+d] = log(X_train[:,d]+1)
	X_train_new[:,2D+d] = X_train[:,d].^2
end 
###### do the same for validation and testing
N_validation = size(X_validation,1)
X_validation_new = zeros(N_validation, D*3)
X_validation_new[1:N_validation, 1:D] = copy(X_validation)
for d=1:D
	X_validation_new[:,D+d] = log(X_validation[:,d]+1)
	X_validation_new[:,2D+d] = X_validation[:,d].^2
end 

N_test = size(X_test,1)
X_test_new = zeros(size(X_test,1), D*3)
X_test_new[1:N_test, 1:D] = copy(X_test)
for d=1:D
	X_test_new[:,D+d] = log(X_test[:,d]+1)
	X_test_new[:,2D+d] = X_test[:,d].^2
end 