function alpha_coefficients_high = polynomial_construct_high(param)

K = max(param.K);
m = param.percentage;
q = sum(param.K)+param.S;

alpha = sdpvar(K+1,1);
gamma = sdpvar(K-m+1,1);

beta = param.beta_coefficients_high;

%% First part of the vector

% For block A
for i = 1:K-m+1
    alpha(i,1) = gamma(1,1) *  beta(i,1);
    for j = 2:i
        alpha(i,1) = alpha(i,1) + gamma(j,1) * beta(i-j+1,1);
    end
end

% For block B
for i = K-m+2:m+1
    alpha(i,1) = gamma(1,1) * beta(i,1);
    for j = 2:K-m+1
        alpha(i,1) = alpha(i,1) + gamma(j,1) * beta(i-j+1,1);
    end
end

% For block C
index = 2;
for i = m+2:K+1
    alpha(i,1) = gamma(index,1) * beta(m+1,1);
    for j = index+1:K-m+1
        alpha(i,1) = alpha(i,1) + gamma(j,1) * beta(m+1-(j-index),1);
    end
    index = index + 1;
end
    alpha_coefficients_high = alpha;
end