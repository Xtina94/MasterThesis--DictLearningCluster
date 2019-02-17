clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset

for trial = 1:1
    %% Loaging the required dataset
    flag = 1;
    switch flag
        case 1
            load ComparisonHeat30.mat
            load DataSetHeat30.mat
        case 2
            load ComparisonDorinaLF.mat
            load DataSetDorinaLF.mat
    end
    
    %% Set the parameters
    
    switch flag
        case 1 %1 LF heat kernel
            param.S = 1;
            param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
            degree = 15;
            param.N = 30;
            ds = 'Dataset used: Synthetic data from Dorina - 1 single kernel';
            ds_name = 'Heat';
            param.percentage = 6;
            param.thresh = param.percentage + 18;
        case 2
            param.S = 1;
            param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
            degree = 20;
            param.N = 30;
            ds = 'Dataset used: Synthetic data from Dorina - 1 single kernel';
            ds_name = 'DorinaLF';
            param.percentage = 13;
            param.thresh = param.percentage + 13;
    end
    
    param.J = param.N * param.S; % total number of atoms
    param.K = degree*ones(1,param.S);
    param.T0 = 4; % sparsity level in the training phase
    param.c = 1; % spectral control parameters
    param.mu = 1e-2; % polynomial regularizer paremeter
    path = ['C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DictionaryLearning\AlphaStructure\Results\18.07.2018\',num2str(ds_name),'\']; %Folder containing the results to save
    
    %------------------------------------------------------------
    %%- Compute the Laplacian and the normalized Laplacian operator
    %------------------------------------------------------------
    
    L = diag(sum(W,2)) - W; % combinatorial Laplacian
    param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
    [param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
    [param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order
    
    
    %%%%%%%
    %% Redistributing the eigenValues composition of the Laplacian
    % % % nEigV = length(param.eigenVal); %Number of eigenvalues
    % % % param.eigenvalues_vector = param.eigenVal*ones(nEigV,1);
    % % % param.eigenvalues_vector = sort(param.eigenvalues_vector);
    % % % eigenVal = param.eigenvalues_vector(1:nEigV-param.percentage);
    % % % eigenVal_1 = eigenVal(1:floor(length(eigenVal)/7));
    % % % section = length(eigenVal_1);
    % % % eigenVal_2 = eigenVal(3*section+1:4*section);
    % % % eigenVal_3 = eigenVal(5*section+1:6*section);
    % % % eigenVal(1:3*section) = [eigenVal_1 eigenVal_1 eigenVal_1];
    % % % eigenVal(3*section+1:5*section) = [eigenVal_2 eigenVal_2];
    % % % i = 1;
    % % % while 6*section+i <= length(eigenVal) && i <= length(eigenVal_3)
    % % %     eigenVal(6*section+i) = eigenVal_3(i);
    % % %     i = i+1;
    % % % end
    % % %
    % % % param.eigenvalues_vector(1:nEigV-param.percentage) = eigenVal;
    % % %
    
    %%%%%%%
    
    %% Analyse the spectrum of the signal
    % spectrum = spectral_rep(param.eigenvalues_vector');
    
    % % % smoothed_signal = smooth_signal(TestSignal, L);
    
    %% Precompute the powers of the Laplacian
    
    for k=0 : max(param.K)
        param.Laplacian_powers{k + 1} = param.Laplacian^k;
    end
    
    for j=1:param.N
        for i=0:max(param.K)
            param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
            param.lambda_power_matrix(j,i + 1) = param.lambda_sym(j)^(i);
        end
    end
    
    param.InitializationMethod =  'Random_kernels';
    
    %%---- Polynomial Dictionary Learning Algorithm -----------
    
    % % % param.initial_dictionary = initial_dictionary;
    param.displayProgress = 1;
    param.numIteration = 20;
    param.plot_kernels = 1; % plot the learned polynomial kernels after each iteration
    param.quadratic = 0; % solve the quadratic program using interior point methods
    
    disp('Starting to train the dictionary');
    
    [Dictionary_Pol, output_Pol]  = Polynomial_Dictionary_Learning(TrainSignal, param);
    
    CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal, param.T0);
    errorTesting_Pol = sqrt(norm(TestSignal - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
    disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);
    
    sum_kernels = sum(output_Pol.g_ker,2);
    
    %% Plot the kernels
    original_ker = zeros(param.N, param.S);
    r = 0;
    for i = 1 : param.S
        for n = 1 : param.N
            p = 0;
            for l = 0 : param.K(i)
                p = p + comp_alpha(l + 1 + r)*param.lambda_powers{n}(l + 1);
            end
            original_ker(n,i) = p;
        end
        r = sum(param.K(1:i)) + i;
    end
    
    figure('Name','learned Kernel VS original')
    subplot(2,1,1)
    hold on
    for s_low = 1 : param.S
        stem(param.lambda_sym(1:length(param.lambda_sym)),output_Pol.g_ker(1:length(param.lambda_sym),s_low));
    end
    hold off
    subplot(2,1,2)
    hold on
    for s_low = 1 : param.S
        stem(param.lambda_sym(1:length(param.lambda_sym)),original_ker(1:length(param.lambda_sym),s_low));
    end
    hold off
    
    filename = [path,'Kernels comparison_trial',num2str(trial)];
    saveas(gcf,filename,'bmp');
    
    %% Plot the alpha coefficients
    figure('Name','The alpha coefficients')
    hold on
    stem(comp_alpha)
    stem(output_Pol.alpha)
    legend('Original alphas','learned alphas')
    hold off
    
    filename = [path,'Coefficients comparison_trial',num2str(trial)];
    saveas(gcf,filename,'bmp');
    
    %% Save results to file
    filename = [path,'Output_results_trial',num2str(trial)];
    totalError = output_Pol.totalError;
    alpha_coeff = output_Pol.alpha;
    g_ker = output_Pol.g_ker;
    save(filename,'Dictionary_Pol','totalError','alpha_coeff', 'g_ker','CoefMatrix_Pol','errorTesting_Pol','TrainSignal','TestSignal','sum_kernels');
end