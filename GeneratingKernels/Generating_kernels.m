%Generate the polynomial approximation of a sin function thorugh its Talor
%expansion
clear all
close all
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\GeneratingKernels\';

syms x
g = 1.5*sin(3*x);
g_6 = cos(2*x);
g_7 = -1.3*sin(3.85*x);
g_8 = 1.17*sin(3.5*x);
g_9 = -2*cos(1.75*x);
g_2(x) = 2.7*sin(4*x) + 0.55*cos(5*x);
g_3(x) = -1.3*sin(3*x) - 2.5*cos(2*x);
g_4(x) = 1.17*sin(3.5*x) - 2*cos(1.75*x);
g_5(x) = 3.5*sin(4.7*(-x) - 3) - 0.55*cos(2.5*(-x) + 3) + 0.6466;

interval = [0,2];
figure('Name','No approx')
hold on
fplot(g_2,interval)
fplot(g_3,interval)
fplot(g_4,interval)
fplot(g_5,interval)
hold off

t = taylor(g, 'Order', 16);
t_2 = taylor(g_2, 'Order', 16);
t_3 = taylor(g_3, 'Order', 16);
t_4 = taylor(g_4, 'Order', 16);
t_5 = taylor(g_5, 'Order', 16);

% t = t + 1.85;
t_2 = (t_2 + 2.21)/7.65;
t_3 = (t_3 + 3.5)/8;
t_4 = (t_4 + 3.5)/8;
t_5 = (t_5 + 3.36)/7.65;

filename = strcat(path,'Original kernels.png');
saveas(gcf,filename,'png');

alpha_2 = sym2poly(t_2);
alpha_3 = sym2poly(t_3);
alpha_4 = sym2poly(t_4);
alpha_5 = sym2poly(t_5);

syms x;
pol = 0;
% for i = 1:16
%     if alpha_2(i) < 10e-4
%         alpha_2(i) = alpha_2(i)*10;
%         pol = pol + (x^(i-1))*alpha_2(i);
%     end
% end

interval = [0,1.3];
figure('Name','First kernels')
hold on
fplot(t_2,interval)
% fplot(pol,interval) 
% fplot(t_3,interval)
% fplot(t_4,interval)
fplot(t_5,interval)
hold off

filename = strcat(path,'2HF_kernels');
save(filename,'alpha_3','alpha_4');

filename = strcat(path,'2LF_kernels');
save(filename,'alpha_2','alpha_5');