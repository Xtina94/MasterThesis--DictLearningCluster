function [alpha, cpuTime, param] = coefficient_update_interior_point(Data,CoefMatrix,param,sdpsolver,g_ker,p)

my_max = zeros(1,param.S);
eval(['N = param.N',num2str(p),';']);
c = param.c;
epsilon = param.epsilon;
mu = param.mu;
S = param.S;
q = sum(param.K)+S;
alpha = sdpvar(q,1);


K = max(param.K);
eval(['Laplacian_powers = param.Laplacian_powers',num2str(p),';']);
eval(['Lambda = param.lambda_power_matrix',num2str(p),';']);
thresh = param.thresh;

BA = kron(eye(S),Lambda);
BB = kron(ones(1,S),Lambda);

Phi = zeros(S*(K+1),1);
for i = 1 : N
         r = 0;
        for s = 1 : S
            for k = 0 : K
                Phi(k + 1 + r,(i - 1)*size(Data,2) + 1 : i*size(Data,2)) = Laplacian_powers{k+1}(i,:)*CoefMatrix((s - 1)*N+1 : s*N,1 : end);
            end
            r = sum(param.K(1 : s)) + s;
        end
end

YPhi = (Phi*(reshape(Data',1,[]))')';
PhiPhiT = Phi*Phi';

la = length(BA*alpha);
lb = length(BB*alpha);

%-----------------------------------------------
% Define the Objective Function
%-----------------------------------------------

X = norm(Data,'fro')^2 - 2*YPhi*alpha + alpha'*(PhiPhiT + mu*eye(size(PhiPhiT,2)))*alpha;

%-----------------------------------------------
% Define Constraints
%----------------------------------------------- 
% The reasoning is:
% If we are in the firs tot cycles of optimization then don't ask for the
% kernels to be also null in some lambdas; at the same time try to
% understand the behavior of these kernels. When a certain amount of
% iterations has passed then try to separate the kernels into high and low
% frequency kernels in order to impose their behavior towards 0
% with respect to their nature

%% Set constraints
% First learn the kernels for a couple of iterations, so that to foresee
% the behavior
if param.big_epoch < 0
    F = (BA*alpha <= c*ones(la,1))...
        + (BA*alpha >= 0.001*epsilon*ones(la,1));%...
% % %         + (BB*alpha <= (c+0.1*epsilon)*ones(lb,1))...
% % %         + (BB*alpha >= ((c-0.1*epsilon)*ones(lb,1)));
else    
    h = eye(param.S);
    h2 = ones(1,param.S);
            
    for j = 1:param.S
        if j ~= 1
            h(j,j) = 0;
            h2(j) = 0;
        end
    end
    %otherwise we're having a low frequency kernel
    B3 = kron(h,Lambda(size(Lambda,1) - param.percentage+1:size(Lambda,1),:));
    B1 = kron(h,Lambda(1:size(Lambda,1) - thresh,:));
    B2 = kron(h2,Lambda(1:size(Lambda,1) - thresh,:));
    
    for i = 2:param.S
        h = eye(param.S);
        h2 = ones(1,param.S);
        for j = 1:param.S
            if j ~= i
                h(j,j) = 0;
                h2(j) = 0;
            end
        end
        
        %it means that we're facing a high frequency kernel
        if mod(i,2) == 0 %it means that we're facing a high frequency kernel
            B3 = B3 + kron(h,Lambda(1:param.percentage,:));
            B1 = B1 + kron(h,Lambda(thresh + 1:size(Lambda,1),:));
            B2 = B2 + kron(h2,Lambda(thresh + 1:size(Lambda,1),:));
        else             %otherwise we're having a low frequency kernel
            B3 = B3 + kron(h,Lambda(size(Lambda,1) - param.percentage+1:size(Lambda,1),:));
            B1 = B1 + kron(h,Lambda(1:size(Lambda,1) - thresh,:));
            B2 = B2 + kron(h2,Lambda(1:size(Lambda,1) - thresh,:));
        end
    end
    
    l3 = length(B3*alpha);
    l1 = length(B1*alpha);
    l2 = length(B2*alpha);

    F = (B1*alpha <= c*ones(l1,1))...
        + (B1*alpha >= 0.001*epsilon*ones(l1,1))...
        + (B3*alpha <= 0.1011)...%0.001*epsilon*ones(l3,1))...
        + (B3*alpha >= 0.0754);%*ones(l3,1));
% % %         + (B2*alpha <= (c+1*epsilon)*ones(l2,1))...
% % %         + (B2*alpha >= (c-1*epsilon)*ones(l2,1))...
end

%---------------------------------------------------------------------
% Solve the SDP using the YALMIP toolbox 
%---------------------------------------------------------------------

if strcmp(sdpsolver,'sedumi')
    diagnostics = optimize(F,X,sdpsettings('solver','sedumi','sedumi.eps',0,'sedumi.maxiter',200));
elseif strcmp(sdpsolver,'sdpt3')
    diagnostics = optimize(F,X,sdpsettings('solver','sdpt3'));
    elseif strcmp(sdpsolver,'mosek')
    diagnostics = optimize(F,X,sdpsettings('solver','mosek'));
    %sdpt3;
else
    error('??? unknown solver');
end

double(X);
param.objective(param.big_epoch) = double(X);
cpuTime = diagnostics.solveroutput.info.cputime;
alpha = double(alpha);
