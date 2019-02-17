clear all
close all

%% Adding the paths
addpath('C:\Users\cryga\Documents\GitHub\DictLearningCluster\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\cryga\Documents\GitHub\DictLearningCluster\DataSets\Comparison_datasets'); %Folder containing the copmarison datasets
addpath('C:\Users\cryga\Documents\GitHub\DictLearningCluster\DataSets'); %Folder containing the training and verification dataset

%% Loading the required dataset
flag = 1;
switch flag
    case 1
        load ComparisonDorina.mat
        load DataSetDorina.mat
    case 2
        load ComparisonHeat30.mat 
        load DataSetHeat30.mat
    case 3
        load ComparisonDoubleHeat.mat
        load DataSetDoubleHeat.mat
    case 4
        load ComparisonUber.mat
        load DataSetUber.mat
end

%% Set the parameters

switch flag
    case 1 %Dorina
        param.S = 4;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        param.N = 100; % number of nodes in the graph
        ds = 'Dataset used: Synthetic data from Dorina';
        ds_name = 'Dorina';
        param.percentage = 15;
        param.thresh = param.percentage+60;
    case 2 %Cristina
        param.S = 1;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 30; % number of nodes in the graph
        ds = 'Dataset used: data from Cristina';
        ds_name = 'Heat'; 
        param.percentage = 8;
        param.thresh = param.percentage+6;
    case 3 %Double heat
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 30; % number of nodes in the graph
        ds = 'Dataset used: data from Cristina';
        ds_name = 'DoubleHeat'; 
        param.percentage = 8;
        param.thresh = param.percentage+6;
        temp = comp_alpha;
        comp_alpha = zeros(param.S*(degree+1),1);
        for i = 1:param.S
            comp_alpha((i-1)*(degree+1)+1:i*(degree+1),1) = temp(:,i);
        end
    case 4 %Uber
        param.S = 2;
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 29; % number of nodes in the graph
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        param.percentage = 8;
        param.thresh = param.percentage+6;
end

path = ['C:\Users\cryga\Documents\GitHub\DictLearningCluster\DictionaryLearning\OriginalAlgo\Results\17.02.19\', num2str(ds_name),'\']; %Folder containing the results to save
% path = '';
param.K = degree*ones(1,param.S);
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.mu = 1e-2; % polynomial regularizer paremeter

%% Compute the Laplacian and the normalized laplacian operator
    
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
% % % [param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
% % % [param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order

%% Find the two communities separation through fiedler's eigenvector
[param, W1, W2, TrainSignal_comm, TestSignal_comm] = comm_det(param, W, TrainSignal, TestSignal);

%% Compute the powers of the Laplacian

for p = 1:2
    for k=0 : max(param.K)
        eval(['param.Laplacian_powers',num2str(p),'{k + 1} = param.Laplacian',num2str(p),'^k;']);
    end
    
    eval(['N = length(param.pos',num2str(p),');']);
    
    for j=1:N
        for i=0:max(param.K)
            eval(['param.lambda_powers',num2str(p),'{j}(i + 1) = param.lambda_sym',num2str(p),'(j)^(i);']);
            eval(['param.lambda_power_matrix',num2str(p),'(j,i + 1) = param.lambda_sym',num2str(p),'(j)^(i);']);
        end
    end
end
    
%% Polynomial dictionary learning algorithm

param.InitializationMethod =  'Random_kernels';
param.displayProgress = 1;
param.numIteration = 20;
param.plot_kernels = 1; % plot thelearned polynomial kernels after each iteration
param.quadratic = 0; % solve the quadratic program using interior point methods

disp('Starting to train the dictionary');

for p = 1:2
    eval(['param.N',num2str(p), ' = length(W',num2str(p),')']);
    eval(['param.J = param.N',num2str(p),' * param.S;']); % total number of atoms
    eval(['[Dictionary_Pol',num2str(p),',output_Pol',num2str(p),']  = Polynomial_Dictionary_Learning(TrainSignal_comm{',num2str(p),'}, param, path,p);']);
end

%% Reconstruct the dictionary
compl1 = zeros(length(param.pos1),length(param.pos2)); %zero padding to make the dimensions match
compl2 = zeros(length(param.pos2),length(param.pos1)); %zero padding to make the dimensions match

tmp = Dictionary_Pol1;
Dictionary_Pol1 = [];
for i = 1:param.S
    Dictionary_Pol1 = [Dictionary_Pol1 tmp(:,(i-1)*length(param.pos1)+1:i*length(param.pos1)) compl1];
end
tmp = Dictionary_Pol2;
Dictionary_Pol2 = [];
for i = 1:param.S
    Dictionary_Pol2 = [Dictionary_Pol2 compl2 tmp(:,(i-1)*length(param.pos2)+1:i*length(param.pos2))];
end

Dictionary_Pol(param.pos1,:) = Dictionary_Pol1;
Dictionary_Pol(param.pos2,:) = Dictionary_Pol2;

%% The Local Error in the two subgraphs
CoefMatrix_Pol1 = OMP_non_normalized_atoms(Dictionary_Pol1,TestSignal_comm{1},param.T0);
errorTesting_Pol1 = sqrt(norm(TestSignal_comm{1} - Dictionary_Pol1*CoefMatrix_Pol1,'fro')^2/size(TestSignal_comm{1},2));
CoefMatrix_Pol2 = OMP_non_normalized_atoms(Dictionary_Pol2,TestSignal_comm{2},param.T0);
errorTesting_Pol2 = sqrt(norm(TestSignal_comm{2} - Dictionary_Pol2*CoefMatrix_Pol2,'fro')^2/size(TestSignal_comm{2},2));

%% The global error in the original graph
CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal,param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%% Compute the l-2 norms

lambda_norm = 'is 0 since here we are learning only the kernels'; %norm(comp_eigenVal - eigenVal);
% alpha_norm = norm(comp_alpha - output_Pol.alpha(1:(param.S - 1)*(degree + 1)));
% X_norm = norm(comp_X - CoefMatrix_Pol(1:(param.S - 1)*param.N,:));
% D_norm = norm(comp_D - Dictionary_Pol(:,1:(param.S - 1)*param.N));
X_norm = norm(comp_X - CoefMatrix_Pol);
D_norm = norm(comp_D - Dictionary_Pol);
alpha_norm1 = norm(comp_alpha - output_Pol1.alpha);
alpha_norm2 = norm(comp_alpha - output_Pol2.alpha);
% % % X_tot_norm = norm([(comp_X - CoefMatrix_Pol) (comp_train_X - output_Pol.CoefMatrix)]);
% % % W_norm = 'is 0 since here we are learning only the kernels';

%% Compute the average CPU_time

avgCPU1 = mean(output_Pol1.cpuTime);
avgCPU2 = mean(output_Pol2.cpuTime);
avgCPU = mean([avgCPU1, avgCPU2]);

%% Save the results to file

% The norms
filename = [path,'Norms.mat'];
save(filename,'lambda_norm','alpha_norm1','alpha_norm2','X_norm','D_norm');

% The Output data
filename = [path,'Output.mat'];
% learned_alpha = output_Pol.alpha;
% save(filename,'ds','Dictionary_Pol','learned_alpha','CoefMatrix_Pol','errorTesting_Pol','avgCPU');
save(filename,'errorTesting_Pol','errorTesting_Pol1','errorTesting_Pol2','avgCPU','avgCPU1','avgCPU2')

% The kernels plot
figure('Name','Final Kernels - 1')
hold on
for s = 1 : param.S
    plot(param.lambda_sym1,output_Pol1.kernel(:,s));
end
hold off

filename = [path,'FinalKernels_plot1.png'];
saveas(gcf,filename);

% The kernels plot
figure('Name','Final Kernels - 2')
hold on
for s = 1 : param.S
    plot(param.lambda_sym2,output_Pol2.kernel(:,s));
end
hold off

filename = [path,'FinalKernels_plot2.png'];
saveas(gcf,filename);

% The CPU1 time plot
xq = 0:0.2:param.numIteration;
figure('Name','CPU time per iteration - 1')
 vq2 = interp1(1:param.numIteration,output_Pol1.cpuTime,xq,'spline');
plot(1:param.numIteration,output_Pol1.cpuTime,'o',xq,vq2,':.');
xlim([0 param.numIteration]);

filename = [path,'AvgCPU_1_timePlot.png'];
saveas(gcf,filename);

% The CPU2 time plot
xq = 0:0.2:param.numIteration;
figure('Name','CPU time per iteration - 2')
vq2 = interp1(1:param.numIteration,output_Pol2.cpuTime,xq,'spline');
plot(1:param.numIteration,output_Pol2.cpuTime,'o',xq,vq2,':.');
xlim([0 param.numIteration]);

filename = [path,'AvgCPU_2_timePlot.png'];
saveas(gcf,filename);

%% Compare the learned coefficients
temp = comp_alpha;
comp_alpha = zeros(degree+1,param.S);
for i = 1:param.S
    comp_alpha(:,i) = temp((degree+1)*(i-1) + 1:(degree+1)*i);
end
for p = 1:2
    eval(['temp = output_Pol',num2str(p),'.alpha;']);
    eval(['final_alpha',num2str(p),' = cell(param.S,1);']);
    for i = 1:param.S
        eval(['final_alpha',num2str(p),'{i} = temp((k+1)*(i-1) + 1:(k+1)*i);']);
    end
    
    figure('Name',['The different behavior of the alpha coefficients - Comm',num2str(p)])
    for j = 1:param.S
        subplot(1,param.S,j)
        title(['Comp ker ',num2str(j),'  VS Learned ker ',num2str(j)]);
        hold on
        stem(comp_alpha(:,j))
        eval(['stem(final_alpha',num2str(p),'{j})']);
        hold off
        legend('comparison kernel','learned kernel');
    end
    
    filename = [path,'AlphaCoeff_Comm',num2str(p),'_comparison_trial.fig'];
    saveas(gcf,filename);
end

