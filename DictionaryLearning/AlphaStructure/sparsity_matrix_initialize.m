% Generating the initial sparsity matrix
function initial_sparsity_mx = sparsity_matrix_initialize(param,Y)

    X = zeros(param.S*param.N,size(Y,2));
    gauss_values = cell(1,size(X,2));
    normalized_gauss_values = cell(1,size(X,2));
    positions = cell(1,size(X,2));
    gauss_values_norm = zeros(1,size(X,2));
    
    for j = 1 : size(X,2)
        gauss_values{1,j} = randn(1,param.T0);
        %normalize the values
        gauss_values_norm(1,j) = sqrt(sum(gauss_values{1,j}.^2));         
        if(gauss_values_norm(1,j) == 0)
            gauss_values_norm(1,j) = 1; 
        end
        normalized_gauss_values{1,j} = gauss_values{1,j} ./ gauss_values_norm(1,j);
        positions{1,j} = randperm(size(X,1),param.T0);
        i = 1;
        while i <= param.T0
            X(positions{1,j}(1,i),j) = normalized_gauss_values{1,j}(1,i);
            i = i + 1;
        end
    end
    
    initial_sparsity_mx = X;
end