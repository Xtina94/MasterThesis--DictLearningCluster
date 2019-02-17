clear all
close all
addpath(genpath(pwd))
path = 'C:\Users\Cristina\Documents\GitHub\DictionaryLearning\LearningPart\Adding the third kernel\Results\';

%__________________________________________________________________________%
                     % ADDING THE THIRD KERNEL VERSION %
%__________________________________________________________________________%
attempt_n = 5;
for attempt_index = 1:attempt_n
    flag = 1;
    switch flag
        case 1
            % For synthetic data
            load testdata.mat
            param.S = 3;  % number of subdictionaries
            param.K = [20 20 20]; % polynomial degree of each subdictionary
            param.percentage = 12;
            param.epsilon = 0.55; %0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
            L = diag(sum(W,2)) - W; % combinatorial Laplacian
            true_Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
        case 2
            % For Uber
            param.S = 3;  % number of subdictionaries
            param.K = [15 15 15]; % polynomial degree of each subdictionary
            param.percentage = 8;
            param.epsilon = 0.2; %0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
            load UberData.mat
            TestSignal = X(:,91:110);
            TrainSignal = X(:,1:90);
            L = diag(sum(learned_W,2)) - learned_W; % combinatorial Laplacian
            true_Laplacian = (diag(sum(learned_W,2)))^(-1/2)*L*(diag(sum(learned_W,2)))^(-1/2); % normalized Laplacian
        case 3
            % For Tikhonov regularization synthetic data
            load TikData.mat
            TestSignal = X_smooth(:,901:1000);
            TrainSignal = X_smooth(:,1:900);
            param.S = 3;  % number of subdictionaries
            param.K = [15 15 15]; % polynomial degree of each subdictionary
            param.percentage = 8;
            param.epsilon = 0.2; %0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
            L = diag(sum(W,2)) - W; % combinatorial Laplacian
            true_Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
        case 4
            % For heat regularization synthetic data
            load HeatData.mat
            TestSignal = X_smooth(:,901:1000);
            TrainSignal = X_smooth(:,1:900);
            param.S = 3;  % number of subdictionaries
            param.K = [15 15 15]; % polynomial degree of each subdictionary
            param.percentage = 8;
            param.epsilon = 0.55; %0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
            L = diag(sum(W,2)) - W; % combinatorial Laplacian
            true_Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
        case 5
            % For no regularization synthetic data
            load NoReg.mat
            TestSignal = X_smooth(:,901:1000);
            TrainSignal = X_smooth(:,1:900);
            param.S = 3;  % number of subdictionaries
            param.K = [15 15 15]; % polynomial degree of each subdictionary
            param.percentage = 8;
            param.epsilon = 0.55; %0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
            L = diag(sum(W,2)) - W; % combinatorial Laplacian
            true_Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
    end
    
    % Common
    X = TrainSignal;
    param.N = size(X,1); % number of nodes in the graph
    param.J = param.N * param.S; % total number of atoms
    K = max(param.K);
    param.c = 1; % spectral control parameters
    param.mu = 1;%1e-2; % polynomial regularizer paremeter
    param.y = X; %signals
    param.y_size = size(param.y,2);
    param.T0 = 6; %sparsity level (# of atoms in each signals representation)
    my_max = zeros(1,param.S);
    
    %% initialise learned data
    
    % Fixed stuff
    [param.Laplacian, learned_W] = init_by_weight(param.N);
    alpha_gradient = 2; %gradient descent parameter, it decreases with epochs
    
    starting_case = 2;
    switch starting_case
        case 1
            % Start from graph learning - (generate dictionary polynomial coefficients from heat kernel)
            param.t = param.K;
            for i = 1 : param.S
                param.t(i) = param.S-(i-1); %heat kernel coefficients; this heat kernel will be inverted to cover high frequency components
            end
            
            param.alpha = generate_coefficients(param);
            
            disp(param.alpha);
            [learned_dictionary, param] = construct_dict(param);
        case 2
            % Start from Dictionary learning - (Initialize the dictionary and the alpha coefficients' structure)
            [param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
            [learned_dictionary] = initialize_dictionary(param);
            [param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order
            [param.beta_coefficients, param.rts] = retrieve_betas(param);
            param.lambda_power_matrix = zeros(param.N,K+1);
            for i = 0:K
                param.lambda_power_matrix(:,i+1) = param.lambda_sym.^i;
            end
            for k = 0 : max(param.K)
                param.Laplacian_powers{k + 1} = param.Laplacian^k;
            end
    end
    
    for big_epoch = 1:20
        %% optimise with regard to x
        disp(['Epoch... ',num2str(big_epoch)]);
        x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
        
        switch starting_case
            case 1
                s = 0;
            case 2
                s = 1;
        end
        
        if mod(big_epoch + s,2) ~= 0
            %optimise with regard to W
            maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
            beta = 10^(-2); %graph sparsity penalty
            old_L = param.Laplacian;
            [param.Laplacian, learned_W] = update_graph(x, alpha_gradient, beta, maxEpoch, param,learned_W, learned_W);
            [learned_dictionary, param] = construct_dict(param);
            alpha_gradient = alpha_gradient*0.985; %gradient descent decreasing
            
            [param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
            [param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order
            param.lambda_power_matrix = zeros(param.N,K+1);
            for i = 0:K
                param.lambda_power_matrix(:,i+1) = param.lambda_sym.^i;
            end
            for k = 0 : max(param.K)
                param.Laplacian_powers{k + 1} = param.Laplacian^k;
            end
        else
            % Set up the elements for the optimization problem
            %Optimize with regard to alpha
            [temp_alpha, my_max] = coefficient_update_interior_point(param.y,x,param,'sdpt3');
            for j = 1:param.S
                param.alpha{j} = temp_alpha((j-1)*(K+1)+1:j*(K+1))';
            end
            [learned_dictionary, param] = construct_dict(param);
        end
        
        %% Plot the kernels
        
        g_ker = zeros(param.N, param.S);
        for i = 1 : param.S
            for n = 1 : param.N
                p = 0;
                for l = 0 : param.K(i)
                    p = p +  param.alpha{i}(l+1)*param.lambda_power_matrix(n,l + 1);
                end
                g_ker(n,i) = p;
            end
        end
        
        param.kernel = g_ker;
        
        if big_epoch == 6
            figure('Name','Last Test on kernel behavior')
            hold on
            for s = 1 : param.S
                plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
            end
            hold off
            
            filename = strcat('trial',num2str(attempt_index),'_',num2str(attempt_n),'a');
            fullfile = [path,filename];
            saveas(gcf,fullfile,'png')
        end
        
    end
    
    figure('Name','Final kernels')
    hold on
    for s = 1 : param.S
        plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
    end
    hold off
    
    filename = strcat('trial',num2str(attempt_index),'_',num2str(attempt_n),'b');
    fullfile = [path,filename];
    saveas(gcf,fullfile,'png');
    
    CoefMatrix_Pol = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
    errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
    disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);
    sum_kernels = sum(param.kernel,2);
    
    %%
    %constructed graph needs to be tresholded, otherwise it's too dense
    %fix the number of desired edges here at nedges
    nedges = 4*29;
    final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
    final_W = learned_W.*(final_Laplacian~=0);
    
    %% Save results to file
    filename = [path,'Output_results_Uber_attempt',num2str(attempt_index),'.mat'];
    alpha_coeff = zeros(K+1,2);
    for i = 1:param.S
        alpha_coeff(:,i) = param.alpha{i};
    end
    save(filename,'final_Laplacian','final_W','alpha_coeff', 'g_ker','CoefMatrix_Pol','errorTesting_Pol','TrainSignal','TestSignal','sum_kernels','learned_dictionary');
    
    
    %% Verifying the results with the precision recall function
    learned_Laplacian = param.Laplacian;
    [optPrec, optRec, opt_Lapl] = precisionRecall(true_Laplacian, learned_Laplacian);
    filename = [path,'ouput_PrecisionRecall_attempt',num2str(attempt_index),'.mat'];
    save(filename,'opt_Lapl','optPrec','optRec');
end