clear all
close all

%% Adding the paths
addpath('C:\Users\cryga\Documents\GitHub\DictLearningCluster\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\cryga\Documents\GitHub\DictLearningCluster\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\cryga\Documents\GitHub\DictLearningCluster\DataSets\'); %Folder containing the training and verification dataset

%% Loading the required dataset
flag = 5;
switch flag
    case 1
        load ComparisonDorina.mat
        load DataSetDorina.mat
        % % %         load Dorina1kernel.mat
    case 2
        load ComparisonHeat30.mat
        load DataSetHeat30.mat
    case 3
        load ComparisonUber.mat
        load DataSetUber.mat
    case 4
        load ComparisonDoubleHeat.mat
        load DataSetDoubleHeat.mat
    case 5
        load ComparisonDorinaLF.mat
        load DataSetDorinaLF.mat
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
    case 2 %1 LF heat kernel
        param.S = 1;
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 30;
        ds = 'Dataset used: Synthetic data from Dorina - 1 single kernel';
        ds_name = 'Heat';
        param.percentage = 8;
        param.thresh = param.percentage + 18;
    case 3 %Uber
        param.S = 2;
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 29; % number of nodes in the graph
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        param.percentage = 8;
        param.thresh = param.percentage+6;
    case 4 %Cristina
        param.S = 2;  % number of subdictionaries
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 30; % number of nodes in the graph
        ds = 'Dataset used: data from double heat kernel';
        ds_name = 'DoubleHeat';
        param.percentage = 8;
        param.thresh = param.percentage+6;
        temp = comp_alpha;
        comp_alpha = zeros((degree+1)*param.S,1);
        for i = 1:2
            comp_alpha((degree+1)*(i-1)+1:(degree+1)*i) = temp(:,i);
        end
    case 5
        param.S = 1;
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        param.N = 30;
        ds = 'Dataset used: Synthetic data from Dorina - 1 single kernel';
        ds_name = 'DorinaLF';
        param.percentage = 15;
        param.thresh = param.percentage + 3;
end

P = 3; % The number of clusters
param.J = param.N * param.S; % total number of atoms
param.K = degree*ones(1,param.S);
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.mu = 1e-2; % polynomial regularizer paremeter
path = ['C:\Users\cryga\Documents\GitHub\DictLearningCluster\DictionaryLearning\Constraints\Results\28.02.2019\',num2str(ds_name),'\']; %Folder containing the results to save

for trial = 1:1
    %% Initialize the kernel coefficients
    temp = comp_alpha;
    comp_alpha = zeros(degree+1,param.S);
    for i = 1:param.S
        comp_alpha(:,i) = temp((degree+1)*(i-1) + 1:(degree+1)*i);
    end
    
    [comp_lambdaSym,comp_indexSym] = sort(diag(comp_eigenVal));
    comp_lambdaPowerMx(:,2) = comp_lambdaSym;
    
    for i = 1:degree+1
        comp_lambdaPowerMx(:,i) = comp_lambdaPowerMx(:,2).^(i-1);
    end
    
    comp_ker = zeros(param.N,param.S);
    for i = 1 : param.S
        for n = 1:param.N
            comp_ker(n,i) = comp_ker(n,i) + comp_lambdaPowerMx(n,:)*comp_alpha(:,i);
        end
    end
    
    %     color = (['g'; 'r'; 'y'; 'b']);
    figure('Name','Original Kernels plot')
    title('Original kernels');
    hold on
    for s = 1 : param.S
        plot(comp_lambdaSym,comp_ker(:,s));
        set(gca,'YLim',[0 1.4])
        set(gca,'YTick',(0:0.2:1.4))
        set(gca,'XLim',[0 1.4])
        set(gca,'XTick',(0:0.2:1.4))
    end
    hold off
    
    %% Compute the Laplacian and the normalized laplacian operator
    L = diag(sum(W,2)) - W; % combinatorial Laplacian
    param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
    
    %% Find the two communities separation through fiedler's eigenvector
    [param, W, TrainSignal_comm, TestSignal_comm, idx] = comm_det_kmn(param, W, TrainSignal, TestSignal,P);
    
    %% Compute the powers of the Laplacian
    
    for p = 1:P
        for k=0 : max(param.K)
            eval(sprintf('param.Laplacian_powers%d{k + 1} = param.Laplacian%d^k;',p,p));
        end
        
        eval(sprintf('N = length(param.tpos{%d});',p));
        
        for j=1:N
            for i=0:max(param.K)
                eval(sprintf('param.lambda_powers%d{j}(i + 1) = param.lambda_sym%d(j)^(i);',p,p));
                eval(sprintf('param.lambda_power_matrix%d(j,i + 1) = param.lambda_sym%d(j)^(i);',p,p));
            end
        end
    end
    
    %% Polynomial dictionary learning algorithm
    
    param.InitializationMethod =  'Random_kernels';
    param.displayProgress = 1;
    param.numIteration = 50;
    param.plot_kernels = 1; % plot thelearned polynomial kernels after each iteration
    param.quadratic = 0; % solve the quadratic program using interior point methods
    
    disp('Starting to train the dictionary');
    
    param.alpha = cell(param.S,1);
    alphas = zeros((degree+1)*param.S,P);
    for p = 1:P
        eval(sprintf('param.percentage = param.myPercentage{%d};', p));
        eval(sprintf('param.N%d = length(W{%d})',p,p));
        eval(sprintf('param.J = param.N%d * param.S;',p)); % total number of atoms
        eval(sprintf('[Dictionary_Pol%d, output_Pol%d]  = Polynomial_Dictionary_Learning(TrainSignal_comm{%d}, param, path,p);',p,p,p));
        
        %% Reconstruct the dictionary
        % Get a Mean on the alpha coefficients of the P communities
        eval(sprintf('alphas(:,p) = output_Pol%d.alpha;',p));
    end
    param.alpha{1}(:) = mean(alphas(1:param.K(1)+1,:),2);
    for i = 2:param.S
        param.alpha{i}(:) = mean(alphas((i-1)*(param.K(i-1)+1)+1:(i-1)*(param.K(i-1)+1)+(param.K(i)+1),:),2);
    end
    
    Dictionary_Pol = construct_dict(param);
    
    %% The Local Error in the two subgraphs
    for p = 1:P
        exp1 = sprintf('CoefMatrix_Pol%d = OMP_non_normalized_atoms(Dictionary_Pol%d,TestSignal_comm{%d},param.T0);',p,p,p);
        exp2 = sprintf('errorTesting_Pol%d = sqrt(norm(TestSignal_comm{%d} - Dictionary_Pol%d*CoefMatrix_Pol%d,''fro'')^2/size(TestSignal_comm{%d},2));', p,p,p,p,p);
        eval(exp1); eval(exp2);
    end
    %% The global error in the original graph
    CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal,param.T0);
    errorTesting_Pol = sqrt(norm(TestSignal - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
    disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);
    
    %% Compute the l-2 norms
    lambda_norm = 'is 0 since here we are learning only the kernels'; %norm(comp_eigenVal - eigenVal);
    avgCPU = zeros(param.numIteration,P);
    for p = 1:P
        %% Compare the learned coefficients
        eval(sprintf('temp = output_Pol%d.alpha;',p));
        eval(sprintf('final_alpha%d = cell(param.S,1);',p));
        for i = 1:param.S
            eval(sprintf('final_alpha%d{i} = temp((k+1)*(i-1) + 1:(k+1)*i);',p));
        end
        
        figure('Name',['The different behavior of the alpha coefficients - Comm',num2str(p)])
        for j = 1:param.S
            subplot(1,param.S,j)
            title(sprintf('Comp ker %d  VS Learned ker %d',j,j));
            hold on
            stem(comp_alpha(:,j))
            eval(sprintf('stem(final_alpha%d{j})',p));
            hold off
            legend('comparison kernel','learned kernel');
        end
        
        filename = [path,'AlphaCoeff_Comm',num2str(p),'_comparison_trial',num2str(trial),'.fig'];
        saveas(gcf,filename);
        
        %% Compute the average CPU_time
        eval(sprintf('avgCPU%d = mean(output_Pol%d.cpuTime);',p,p));
        eval(sprintf('avgCPU(:,p) = avgCPU%d;',p));
    end
    avgCPU = mean(mean(avgCPU,2));
    
    %% Save the results to file
    for p = 1:P
        % The Output data
        learned_alpha = zeros(degree+1,param.S);
        for i = 1:param.S
            learned_alpha(:,i) = param.alpha{i};
        end
        eval(sprintf('alpha%d = output_Pol%d.alpha;',p,p));
        name = sprintf('Output%d.mat', p);
        filename = strcat(path,'\',name);
        var1 = sprintf('avgCPU%d',p); var2 = sprintf('alpha%d',p); var3 = sprintf('errorTesting_Pol%d',p);
        save(filename,'ds','learned_alpha','errorTesting_Pol','avgCPU',var1,var2,var3);
    end
end
% The kernels plot
for p = 1:P
    figure('Name',['Comparison between the Kernels - Comm',num2str(p)])
    subplot(2,1,1)
    title('Original kernels');
    hold on
    for s = 1 : param.S
        plot(comp_lambdaSym,comp_ker(:,s));
        set(gca,'YLim',[0 1.4])
        set(gca,'YTick',(0:0.2:1.4))
        set(gca,'XLim',[0 1.4])
        set(gca,'XTick',(0:0.2:1.4))
    end
    hold off
    subplot(2,1,2)
    title('learned kernels');
    hold on
    for s = 1 : param.S
        eval(sprintf('plot(param.lambda_sym%d,output_Pol%d.kernel(:,s));',p,p));
        set(gca,'YLim',[0 1.4])
        set(gca,'YTick',(0:0.2:1.4))
        set(gca,'XLim',[0 1.4])
        set(gca,'XTick',(0:0.2:1.4))
    end
    hold off
    filename = [path,'FinalKernels_comm',num2str(p),'_plot_trial',num2str(trial),'.png'];
    saveas(gcf,filename);
end