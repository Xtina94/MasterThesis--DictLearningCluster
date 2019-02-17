function [beta_coefficients, rts] = retrieve_betas(param)
% I have to initialize the beta coefficients in order to have a reduced
% polynomial to learn. I don't have to initialize also the gamma
% coefficients, I have to retrieve those ones from the coefficient
% update step

%Input:
%     percentage = threshold for filtering the kernels. It has to be a value 0<= m <= K with K = degree of polynomial.
%     and indicates the number of eigenvalues that are the roots of our
%     polynomial kernel
%     param-lambda_sym = vector containing all the eigenvalues
       
    %% building up the transpose vandermonde matrix of the eigenvalues
    m = param.percentage;
    vand_eig = zeros(m,m+1);
    img_dim = m+1;
    lambdas = param.lambda_sym;
    
    while img_dim >= m
        rts =  lambdas(length(lambdas)-m+1:length(lambdas),1); 
        for i = 1:m+1
            for j = 1:m
                vand_eig(j,i) = rts(j)^(i-1);
            end
        end
        m = m+1;
        img_dim = rank(vand_eig);
    end
    
    m = m - 1;
    
    %% retrieving the kernel of the Vandermonde matrix
    
    betas = null(vand_eig);    
    i = 1;
    
    if betas(1,1) > 0
        beta_coefficients = betas(:,1);
    else
        while betas(1,i) < 0
            if i < length(betas(1,:))
                i = i+1;
            elseif i == length(betas(1,:))
                betas = -betas;
            end
        end
        beta_coefficients = (betas(:,i));
    end 
end