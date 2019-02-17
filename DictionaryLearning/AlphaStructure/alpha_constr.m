function alpha = alpha_constr(param)
K = max(param.K);
if param.percentage >= K/2
    M = param.percentage;
    N = K - M;
    alpha_1 = sdpvar(N,1);
    alpha_2 = sdpvar(K+1-2*N,1);
    alpha_3 = sdpvar(N,1);
    gamma = sdpvar(N+1,1);
    beta = param.beta_coefficients_low'; % Beta is a vector in the form: "beta = [b0, b1, b2, ..., bm]"
else
    N = param.percentage;
    M = K - N;
    alpha_1 = sdpvar(N,1);
    alpha_2 = sdpvar(K+1-2*N,1);
    alpha_3 = sdpvar(N,1);
    beta = sdpvar(1,M+1);
    gamma = param.beta_coefficients_low; % Beta is a vector in the form: "beta = [b0, b1, b2, ..., bm]"
end

% q = sum(param.K)+param.S;
% alpha_1 = zeros(N,1);
% alpha_2 = zeros(K+1-2*N,1);
% alpha_3 = zeros(N,1);
% gamma = [1:N+1]';

%% First part of the vector

% For block A
for i = 1:N
    alpha_1(i) = beta(1:i)*flipud(gamma(1:i));
end

% For block B
b = K+1-2*N;
for i = 1:b
    alpha_2(i) = beta(i:i+N)*flipud(gamma);
end

%For block C
for i = 1:N
    alpha_3(i) = beta(b+i:M+1)*flipud(gamma(i+1:N+1));
end

alpha = [alpha_1; alpha_2; alpha_3];

% For block C
% for i = N+2:M+1
%     for j = 1:N+1
%         alpha(i) = alpha(i) + beta(i-(N+1)+j)*gamma(N+2-j);
%     end
% end

% For block D
% l = 0;
% for i = M+2:K+1
%     for j = 1+l:N
%         alpha(i) = alpha(i) + beta(i-(N+1+l-j))*gamma(j+1);
%     end
%     l = l + 1;
% end

end
