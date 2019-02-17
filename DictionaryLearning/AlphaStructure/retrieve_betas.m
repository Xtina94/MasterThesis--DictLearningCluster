function [beta_coefficients_low, beta_coefficients_high, rts_low, rts_high] = retrieve_betas(param)
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
    vand_eig_low = zeros(m,m+1);
    vand_eig_high = zeros(m,m+1);
    img_dim_low = m+1;
    img_dim_high = m+1;
    lambdas = param.lambda_sym;
    
    while img_dim_low >= m || img_dim_high >= m
        rts_low = lambdas(length(lambdas)-m+1:length(lambdas),1);
        rts_high = lambdas(1:m,1);        
        for i = 1:m+1
            for j = 1:m
                vand_eig_low(j,i) = rts_low(j)^(i-1);
                vand_eig_high(j,i) = rts_high(j)^(i-1);
            end
        end
        m = m+1;
        img_dim_low = rank(vand_eig_low);
        img_dim_high = rank(vand_eig_high);
    end
    
    m = m - 1;
    
    %% retrieving the kernel of the Vandermonde matrix
    
    betas_low = null(vand_eig_low);
    betas_high = null(vand_eig_high);
    
    i = 1;
    
    if betas_low(1,1) > 0
        beta_coefficients_low = betas_low(:,1);
    else
        while betas_low(1,i) < 0
            if i < length(betas_low(1,:))
                i = i+1;
            elseif i == length(betas_low(1,:))
                betas_low = -betas_low;
            end
        end
        beta_coefficients_low = (betas_low(:,i));
    end 
    
    i = 1;
    
    if betas_high(1,1) > 0
        beta_coefficients_high = betas_high(:,1);
    else
        while betas_high(1,i) < 0
            if i < length(betas_high(1,:))
                i = i+1;
            elseif i == length(betas_high(1,:))
                betas_high = -betas_high;
            end
        end
        beta_coefficients_high = (betas_high(:,i));
    end
end