clear all
close all

%% Adding the paths
addpath('C:\Users\cryga\Documents\GitHub\DictLearningCluster\DataSets\'); %Folder containing the yalmip tools
path = ('C:\Users\cryga\Documents\GitHub\DictLearningCluster\DataSets\Comparison_datasets\');

%% Loading the required dataset
flag = 1;
switch flag
    case 1
        load testdata.mat
        load comp_alpha.mat
        Y_comp = TrainSignal;
        clear 'A' 
        clear 'C'
        clear 'D'
        clear 'TrainSignal'
        clear 'XCoords'
        clear 'YCoords'
%         load ComparisonDorina.mat
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
        param.alpha = cell(param.S,1);
        for s = 1:param.S
            param.alpha{s} = comp_alpha((s-1)*(degree+1)+1:s*(degree+1));
        end
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

for j=1:param.N
    for i=0:max(param.K)
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
        param.lambda_power_matrix(j,i + 1) = param.lambda_sym(j)^(i);
     end
end

% % % % The kernels plot
% % % g_ker = zeros(param.N, param.S);
% % % r = 0;
% % % for i = 1 : param.S
% % %     for n = 1 : param.N
% % %         p = 0;
% % %         for l = 0 : param.K(i)
% % %             p = p +  comp_alpha(l + 1 + r)*param.lambda_powers{n}(l+1);
% % %         end
% % %         g_ker(n,i) = p;
% % %     end
% % %     r = sum(param.K(1:i)) + i;
% % % end
% % % color = ['r';'g';'b';'y'];
% % % figure('Name','Comparison Kernels')
% % % hold on
% % % for s = 1 : param.S
% % %     plot(param.lambda_sym,g_ker(:,s),color(s));
% % % end
% % % hold off

%% Construct the dictionary
[comp_Dictionary_Pol, param] = construct_dict(param);
    
%% Generate the sparsity matrix
for epoch = 1:30
    comp_CoefMatrix = OMP_non_normalized_atoms(comp_Dictionary_Pol,Y_comp, param.T0);
    Y_comp = comp_Dictionary_Pol*comp_CoefMatrix;
end

TrainSignal = Y_comp;
comp_alpha = param.alpha{1};
comp_eigenVal = param.eigenVal;

%Save dataset
filename = [path,'Dorina1kernel.mat'];
save(filename,'comp_eigenVal','W','TrainSignal','comp_Dictionary_Pol','comp_CoefMatrix','comp_alpha','TestSignal');
