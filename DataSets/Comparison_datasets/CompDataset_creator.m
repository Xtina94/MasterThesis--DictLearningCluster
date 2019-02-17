clear all

%% Adding the paths
addpath('C:\Users\cryga\Documents\GitHub\DictLearningCluster\DataSets\'); %Folder containing the yalmip tools
path = ('C:\Users\cryga\Documents\GitHub\DictLearningCluster\DataSets\Comparison_datasets\');

%% Loading the required dataset
flag = 1;
switch flag
    case 1
        load testdata.mat
        ds = 'Dataset used: Synthetic data from Dorina';
        ds_name = 'Dorina';
    case 2
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        load DataSetUber.mat
end

%% Set the parameters

switch flag
    case 1 %Dorina
        param.S = 1;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        param.N = 100; % number of nodes in the graph
    case 2 %Uber
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 29; % number of nodes in the graph
end

param.J = param.N * param.S; % total number of atoms 
param.K = degree*ones(1,param.S);
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.mu = 1e-2; % polynomial regularizer paremeter

%% Compute the Laplacian and the normalized laplacian operator
    
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order

%% Compute the powers of the Laplacian

for k=0 : max(param.K)
    param.Laplacian_powers{k + 1} = param.Laplacian^k;
end
    
for j=1:param.N
    for i=0:max(param.K)
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
        param.lambda_power_matrix(j,i + 1) = param.lambda_sym(j)^(i);
     end
end
    
%% Polynomial dictionary learning algorithm

param.InitializationMethod =  'Random_kernels';
param.displayProgress = 1;
param.numIteration = 50;
param.plot_kernels = 1; % plot thelearned polynomial kernels after each iteration
param.quadratic = 0; % solve the quadratic program using interior point methods

disp('Starting to train the dictionary');

[Dictionary_Pol,output_Pol,g_ker]  = Polynomial_Dictionary_Learning(TrainSignal, param);

CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%% Save the results to file

% The Output data
comp_train_X = output_Pol.CoefMatrix;
comp_X = CoefMatrix_Pol;
comp_W = W;
comp_Laplacian = param.Laplacian;
comp_D = Dictionary_Pol;
comp_eigenVal = param.eigenVal;
comp_alpha = output_Pol.alpha;

% The kernels plot
figure('Name','Comparison Kernels')
hold on
for s = 1 : param.S
    plot(param.lambda_sym,g_ker(:,s));
end
hold off

filename = [path,'Comp_kernels_',num2str(ds_name),'1kernel.png'];
saveas(gcf,filename);

% lf_alpha = comp_alpha((degree+1)*2+1:(degree+1)*3);
filename = [path,'Comparison',num2str(ds_name),'.mat'];
save(filename,'comp_X','comp_W','comp_Laplacian','comp_D','comp_eigenVal','comp_alpha','comp_train_X');

figure()
surf(comp_train_X)
