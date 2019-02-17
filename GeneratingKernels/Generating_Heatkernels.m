%Generate the polynomial approximation of a sin function thorugh its Talor
%expansion
clear all
close all
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\GeneratingKernels\Results\';

param.S = 1;
deg = 15;
param.K = deg*ones(1,param.S);
syms x;

%% generate dictionary polynomial coefficients from heat kernel

for i = 1:param.S
    if mod(i,2) ~= 0
        param.t(i) = 2; %heat kernel coefficients
    else
        param.t(i) = 1; % Inverse of the heat kernel coefficients
    end
end

temp = heat_expansion(param);

for i = 1:param.S
    param.alpha(i,:) = temp{i};
end

syms kernel;

for i = 1:param.S
    eval(strcat('kernel_',num2str(i),'(x) = x^(0)*param.alpha(',num2str(i),',1)'));
end

    for j = 2:deg+1
        kernel_1(x) = kernel_1(x) + x^(j-1)*param.alpha(1,j);
%         kernel_2(x) = kernel_2(x) + x^(j-1)*param.alpha(2,j);
    end

lambdas = [0:0.1:1.5]';
for i = 1:param.S
    eval(strcat('kernels(:,',num2str(i),') = kernel_',num2str(i),'(lambdas)'));
end

lambdas_matrix = zeros(deg+1,deg+1);
for i = 0:deg
    lambdas_matrix(i+1,:) = lambdas(1:deg+1,1)'.^i;
end

kernels(:,2) = flipud(kernels);
param.alpha(2,:) = kernels(1:deg+1,2)'/lambdas_matrix;

load comp_lambdaSym.mat;
temp = sort(lambdas);
lambdas = temp;
lambdas_matrix = zeros(deg+1,length(lambdas));
for i = 0:deg
    lambdas_matrix(i+1,:) = lambdas(:,1)'.^i;
end
kernels = zeros(length(lambdas),2);
kernels(:,2) = (param.alpha(2,:)*lambdas_matrix)';
kernels(:,1) = kernel_1(lambdas);

figure('Name','Heat kernels representation')
hold on
plot(lambdas,kernels(:,1));
set(gca,'XLim',[0 1.4])
set(gca,'XTick',(0:0.2:1.4))
hold off

%% Save results to file

filename = [path,'DoubleInv_HeatKernel_plot.png'];
saveas(gcf,filename);

DoubleInv_HeatKernel = param.alpha;
filename = [path,'DoubleInv_HeatKernel.mat'];
save(filename,'DoubleInv_HeatKernel');

% filename = [path,'DoubleHeatKernel_plot.png'];
% saveas(gcf,filename);
% 
% HeatKernel = param.alpha';
% filename = [path,'HeatKernel.mat'];
% save(filename,'HeatKernel');


