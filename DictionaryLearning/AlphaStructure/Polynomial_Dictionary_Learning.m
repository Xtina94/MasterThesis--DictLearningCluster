function [recovered_Dictionary,output] = Polynomial_Dictionary_Learning(Y, param)

lambda_powers = param.lambda_powers;
Laplacian_powers = param.Laplacian_powers;

if (~isfield(param,'displayProgress'))
    param.displayProgress = 0;
end

if (~isfield(param,'quadratic'))
    param.quadratic = 0;
end

if (~isfield(param,'plot_kernels'))
    param.plot_kernels = 0;
end

if (~isfield(param,'numIteration'))
    param.numIteration = 100;
end

if (~isfield(param,'InitializationMethod'))
% % %     param.InitializationMethod = 'Random_kernels';
    param.InitializationMethod = 'GivenMatrix';
end

color_matrix = ['b', 'r', 'g', 'c', 'm', 'k', 'y'];
 
%%-----------------------------------------------
%% Initializing the dictionary
%%-----------------------------------------------

if (strcmp(param.InitializationMethod,'Random_kernels')) 
    Dictionary(:,1 : param.J) = initialize_dictionary(param);
% % %     initial_dictionary_uber = Dictionary;
        
elseif (strcmp(param.InitializationMethod,'Heat_kernels'))
     Dictionary = param.generate_coefficients(param);
     
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
     Dictionary = param.initial_dictionary;
else 
    display('Initialization method is not valid')
end


%%----------------------------------------------------
%%  Graph Dictionary Learning Algorithm
%%----------------------------------------------------

%% Obtaining the beta coefficients
[param.beta_coefficients_low, param.beta_coefficients_high, param.rts_low, param.rts_high] = retrieve_betas(param);

%% Set up the elements for the optimization problem
n_low = 1;
n_high = param.S - n_low;
K = max(param.K);

for i = 1:n_low
% % %     param.alpha_vector_low((i-1)*(K+1)+1:i*(K+1),1) = polynomial_construct_lf(param);
    param.alpha_vector_low((i-1)*(K+1)+1:i*(K+1),1) = alpha_constr(param);
% % %     param.alpha_vector_low((i-1)*(K+1)+1:i*(K+1),1) = polynomial_construct_low(param);
end
if n_high > 0
    for i = 1:n_high
        param.alpha_vector_high((i-1)*(K+1)+1:i*(K+1),1) = polynomial_construct_high(param);
    end
end

%% Verify the correctness of the constructed vector
% % % product_lambda = param.lambda_power_matrix*param.alpha_vector_low;

for iterNum = 1 : param.numIteration
    
    %%  Sparse Coding Step (OMP)
    CoefMatrix = OMP_non_normalized_atoms(Dictionary,Y, param.T0);
      
    %% Coefficients update step
          
       if (param.quadratic == 0)
           if (iterNum == 1)
            disp('solving the quadratic problem with YALMIP...')
           end
            [alpha, objective_low, objective_high] = coefficient_update_interior_point(Y,CoefMatrix,param,n_low,n_high,'sdpt3');
       else
           if (iterNum == 1)
            disp('solving the quadratic problem with ADMM...')
           end
            [Q1,Q2, B, h] = compute_ADMM_entries(Y, param, Laplacian_powers, CoefMatrix);
             alpha = coefficient_upadate_ADMM(Q1, Q2, B, h);
       end
       
% % %        alpha_mx_low = zeros(K+1,param.S);
% % %        alpha_mx_high = zeros(K+1,param.S);
% % %        
% % %        for s_low = floor(param.S/2) : param.S
% % %            alpha_mx_low(:,s_low) = alpha_low((s_low-1)*(K+1)+1:s_low*(K+1));
% % %        end
% % %        
% % %        for s_high = 1 : floor(param.S/2)
% % %            alpha_mx_high(:,s_high) = alpha_high((s_high-1)*(K+1)+1:s_high*(K+1));
% % %        end

        if (param.plot_kernels == 1) 
            g_ker = zeros(param.N, param.S);
            r = 0;
            for i = 1 : param.S
                for n = 1 : param.N
                p = 0;
                for l = 0 : param.K(i)
                    p = p +  alpha(l + 1 + r)*lambda_powers{n}(l + 1);
                end
                g_ker(n,i) = p;
                end
                r = sum(param.K(1:i)) + i;
            end
            
% % %               figure()
% % %               hold on
% % %               for s = 1 : param.S
% % %                   plot(lambda_sym,g_ker(:,s),num2str(color_matrix(s)));
% % %               end
% % %               hold off
        end

        %% Dictionary update step
        
        r = 0;
        for j = 1 : param.S
            D = zeros(param.N);
            for ii = 0 : param.K(j)
                D = D +  alpha(ii + 1 + r) * Laplacian_powers{ii + 1};
            end
            r = sum(param.K(1:j)) + j;
            Dictionary(:,1 + (j - 1) * param.N : j * param.N) = D;
        end
        
% % %         Dictionary = 1e4*Dictionary;
        
    %% Plot the progress

    if (iterNum>1 && param.displayProgress)
             output.totalError(iterNum - 1) = sqrt(sum(sum((Y-Dictionary * CoefMatrix).^2))/numel(Y));
             disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalError(iterNum-1))]);
    end
    
end

output.g_ker = g_ker;
output.objective_low = objective_low;
output.objective_high = objective_high;
output.CoefMatrix = CoefMatrix;
output.alpha =  alpha;
recovered_Dictionary = Dictionary;