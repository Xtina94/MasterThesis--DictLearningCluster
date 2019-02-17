clear all
close all
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\GeneratingKernels\Results\');
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\';

flag = 3;
switch flag
    case 1
        load '2LF_kernels.mat'
        comp_alpha(:,1) = alpha_2;
        comp_alpha(:,2) = alpha_5;
        kernels_type = 'LF';
        n_kernels = 2;
        k = 15; %polynomial degree        
    case 2
        load '2HF_kernels.mat'
        load 'Output_results.mat'
        alpha_1 = alpha_coeff(:,1);
        alpha_2 = alpha_coeff(:,2);
        kernels_type = 'HF';
        n_kernels = 2;
        k = 15; %polynomial degree
    case 3
        load 'LF_heatKernel.mat'
        n_kernels = 1;
        k = 15; %polynomial degree
        comp_alpha = zeros(k+1,n_kernels);
        for i = 1:n_kernels
            eval(['comp_alpha(:,',num2str(i),') = LF_HeatKernel(:,',num2str(i),')']);
        end
        kernels_type = 'Heat30';
    case 4
        load 'HeatKernel.mat'
        n_kernels = 2;
        k = 15; %polynomial degree
        comp_alpha = zeros(k+1,n_kernels);
        for i = 1:n_kernels
            eval(['comp_alpha(:,',num2str(i),') = HeatKernel(:,',num2str(i),')']);
        end
        kernels_type = 'DoubleHeat';
    case 5
        load 'DoubleInv_HeatKernel.mat'
        n_kernels = 2;
        k = 15; %polynomial degree
        comp_alpha = zeros(k+1,n_kernels);
        for i = 1:n_kernels
            eval(['comp_alpha(:,',num2str(i),') = DoubleInv_HeatKernel(',num2str(i),',:)']);
        end
        kernels_type = 'DoubleInvHeat';
    case 6
        load DorinaLF_coeff.mat
        n_kernels = 1;
        k = 20;
        kernels_type = 'DorinaLF';
end

if flag == 8
    comp_X = comp_X(201:300,:);
    comp_train_X = comp_train_X(201:300,:);
    comp_D = comp_D(:,201:300);
    filename = strcat(path,'Comparison_datasets\Comparison',num2str(kernels_type),'.mat');
    save(filename,'comp_alpha','comp_D','comp_X','comp_train_X','comp_W','comp_Laplacian','comp_eigenVal');
    load DataSetDorina.mat
    filename = strcat(path,'DataSet',num2str(kernels_type),'.mat');
    save(filename,'TestSignal','TrainSignal','W','XCoords','YCoords');
else
    m = 30;
    l = 2000;
    t0 = 4;
    
    %% Obtaining the corresponding weight and Laplacian matrices + the eigen decomposition parameters
    
    uniform_values = unifrnd(0,1,[1,m]);
    sigma = 0.2;
    [W] = random_geometric(sigma,m,uniform_values,0.6);
    L = diag(sum(W,2)) - W;
    Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2);
    
    [eigenVect, comp_eigenVal] = eig(Laplacian);
    [lambda_sym,index_sym] = sort(diag(comp_eigenVal));
    
    %% Plot the kernel
    syms kernel;
    syms x;
    for i = 1:n_kernels
        eval(['kernel_',num2str(i),'(x) = x^(0)*comp_alpha(1,',num2str(i),')']);        
        for j = 2:k+1
            eval(strcat('kernel_',num2str(i),'(x) = kernel_',num2str(i),'(x) + x^(',num2str(j-1),')*comp_alpha(',num2str(j),',1)'));
        end
    end
    
    figure('Name','Kernels plot')
    hold on
    for i = 1:n_kernels
        eval(['fplot(kernel_',num2str(i),',[0,1.35])']);
    end
    hold off
    
    %% Precompute the powers of the Laplacian
    
    Laplacian_powers = cell(1,k+1);
    
    for j = 0 : k
        Laplacian_powers{j + 1} = Laplacian^j;
    end
    
    %% Construct the dictionary
    
    D = cell(1,n_kernels);
    comp_D = zeros(m,n_kernels*m);
    for i = 1:n_kernels
        D{i} = Laplacian_powers{1}*comp_alpha(1,i);
        for j = 2:k+1
            D{i} = D{i} + Laplacian_powers{j}*comp_alpha(j,i);
        end
        comp_D(:,(i-1)*m + 1:i*m) = D{i};
    end
    
    %% Generate the sparsity matrix
    
    X = Generate_sparsity(n_kernels,t0,m,l);
    
    %% Generate the signal through Y = DX
    Y = comp_D*X;
    TrainSignal = Y(:,1:800);
    TestSignal = Y(:,801:2000);
    comp_X = X(:,801:2000);
    comp_train_X = X(:,1:800);
    filename = strcat(path,'DataSet',num2str(kernels_type),'.mat');
    save(filename,'TestSignal','TrainSignal','W');
    
    comp_W = W;
    comp_Laplacian = Laplacian;
    
    %% Save the results to file
    
    filename = strcat(path,'Comparison_datasets\Comparison',num2str(kernels_type),'.mat');
    save(filename,'comp_alpha','comp_D','comp_X','comp_train_X','comp_W','comp_Laplacian','comp_eigenVal');
end