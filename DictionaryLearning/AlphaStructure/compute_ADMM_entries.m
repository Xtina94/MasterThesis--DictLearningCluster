function [Q1,Q2, B, h] = compute_ADMM_entries(Y, param, Laplacian_powers, X)
%======================================================
   %% Compute the required entries for ADMM
%======================================================
% Description: Find Q1, Q2, B, h such that the quadratic program is
% expressed as: 
%       minimize     (1/2)*alpha'*Q1*alpha-Q2*alpha
%       subject to   B*alpha<=h
%======================================================

N = param.N;
c = param.c;
epsilon = param.epsilon;
mu = param.mu;
S = param.S;
K = max(param.K);
Lambda = param.lambda_power_matrix;



B1 = sparse(kron(eye(S),Lambda));
B2 = kron(ones(1,S),Lambda);

Phi = zeros(S*(K+1),1);
for i = 1 : N
         r = 0;
        for s = 1 : S
            for k = 0 : K
                Phi(k + 1 + r,(i - 1)*size(Y,2) +1 : i*size(Y,2)) = Laplacian_powers{k + 1}(i,:)*X((s-1)*N + 1 : s*N,1 : end);
            end
            r = sum(param.K(1 : s)) + s;
        end
end

YPhi = (Phi*(reshape(Y',1,[]))')';
PhiPhiT = Phi*Phi';

Q2 = YPhi;
Q1 = PhiPhiT + mu*eye(size(PhiPhiT,2));


B = [B1; -B1; B2; -B2];
h = [c*ones(size(B1,1),1);zeros(size(B1,1),1);(c + epsilon)*ones(size(B2,1),1); -(c - epsilon)*ones(size(B2,1),1)];