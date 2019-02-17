function smoothed_signal = smooth_signal(TestSignal, laplacian)
% Function used to make the test signal a smooth signal through the
% optimization of the signal itself. adj_matrix is the adjacency matrix of
% the graph

% Parameters
f = sdpvar(100,2000);
L = laplacian;

%% Define the objective function

X = f'*L*f;

%% Set the optimization problem

diagnostic = optimize([],X);

smoothed_signal = f;
end