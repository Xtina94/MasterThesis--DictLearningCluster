close all;
clear all;
load 2LF_kernels.mat;

K = 15;
N = 30;
x = 0:1/K:1;
syms y
y = sin(x*2*pi);

% lambda = sort(1.5*rand(1,N));
lambda = 0:0.0424:1.23;

lambdaMx = zeros(K+1,length(lambda));
for i = 1:K+1
    lambdaMx(i,:) = lambda.^(i-1);
end

figure()
hold on
plot(alpha_2)
plot(alpha_5)
hold off

g_ker{1} = alpha_2*lambdaMx;
g_ker{2} = alpha_5*lambdaMx;

figure()
hold on
plot(g_ker{1})
plot(g_ker{2})
hold off
