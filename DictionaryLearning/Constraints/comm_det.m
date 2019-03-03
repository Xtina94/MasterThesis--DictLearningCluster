function [param, W_comm, TrainSignal, TestSignal] = comm_det(param, W, TRS, TTS,P)
    %%%%%%%%%%% BE CAREFUL: this method is different from the one in
    %%%%%%%%%%% graph leraning! %%%%%%%%%%
    %%%%%%%%
    % Parameters:
    % TRS = original Trainsignal
    % TTS = original TestSignal
    % Trainsignal = The subspace of the train signal composed by only the vertices belonging to
    % the cluster we found
    % TestSignal = The subspace of the test signal composed by only the vertices belonging to
    % the cluster we found
    %%%%%%%%
    
    % extract eigenvectors
    N = param.N;
    d = full(sum(W));
    Di = spdiags(1./sqrt(d'),0,N,N);
    % Eigendecomposition
    [V,DD] = eig(param.Laplacian);
    % Max eigengap
    [DD, pos] = sort(diag(DD));
    gap = DD(1:end-1) - DD(2:end);
    [~, gapPos] = max(gap); % The maximum gap position identifies the number of 
                            % eigenvalues/vectors to be taken to span the
                            % subspace and so the number of centroids
                            
%     clustN = gapPos - 1; % Too many centroids, I need 3 for now
    clustN = P;
    W_comm = cell(clustN,1);
    TrainSignal = cell(clustN,1);
    TestSignal = cell(clustN,1);
    tpos = cell(clustN,1); % The cell matrix containing in each cell the list 
                           % of nodes belonging to cluster i
    
    % Take the clustN smallest eigenvalues and corresponding eigenvectors
    X = V(:,pos(2:clustN+1));
    % Normalize the rows of X
    for i = 1:N
        X(i,:) = X(i,:)./norm(X(i,:));
    end
    
    % Plot the data with coordinates in X
    if P == 3
        figure('Name','The spectral clusters')
        scatter3(X(1,1),X(1,2),X(1,3),'.');
        hold on
        for i = 2:N
            scatter3(X(i,1),X(i,2),X(i,3),'.');
        end
        hold off
    end
    
    cl = 3;
    switch cl
        case 1
            %% Clustering algorithm: PageRank-Nibble approach
            % Initialization step
            stV = 4; p = zeros(N,1); r = zeros(N,1); r(stV) = 1;
            Phi = 0.5; % The conductance imposed
            m = sum(sum(W > 0))/2; % number of edges in the graph
            a = Phi^2/(225*log(100*sqrt(m)));
            B = floor(log2(m));
            b = randi(B);
            vol_low = 2^(b-1);% Lower bound in volume
            vol_up =2/3*(2*m); % upper bound in volume
            p_ch = 1/(48*B ); % probability change
            c = 0.85; %damping factor
            epsilon = 2^(-b)*p_ch; %Trial on the bound
            
            % Push operation
            d = d';
            myMax = max(r./d);
            while (myMax >= epsilon)
                for u = 1:N
                    if (r(u)/d(u) >= epsilon)
                        p(u) = p(u) + a*r(u);
                        r(u) = (1-a)*r(u)/2;
                        for v = 1:N
                            if ismember(v,find(W(u,:)))
                                r(v) = r(v) + (1-a)*r(u)/(2*d(u));
                            end
                        end
                    end
                end
                myMax = max(r./d);
            end
            
        case 2
            %% Clustering algorithm: k-means approach
            idx = kmeans(X,clustN);
            
            for p = 1:clustN
                tmp = (idx == p);
                param.tpos{p} = find(tmp);
                csize = sum(tmp);
                W_comm{p} = W(param.tpos{p},param.tpos{p});
                TrainSignal{p} = TRS(param.tpos{p},param.tpos{p});
                TestSignal{p} = TTS(param.tpos{p},param.tpos{p});
                disp(['Community size #',num2str(p),': ', num2str(csize)]); 
                
                exp1 = sprintf('param.L = diag(sum(W_comm{%d},2)) - W_comm{%d};',p,p);
                exp2 = sprintf('param.Laplacian%d = (diag(sum(W_comm{%d},2)))^(-1/2)*param.L*(diag(sum(W_comm{%d},2)))^(-1/2);', p,p,p);
                exp3 = sprintf('[param.eigenMat%d, param.eigenVal%d] = eig(param.Laplacian%d);', p,p,p);
                exp4 = sprintf('[param.lambda_sym%d, index_sym%d] = sort(diag(param.eigenVal%d));', p,p,p);
                eval(exp1); eval(exp2); eval(exp3); eval(exp4);
                eval(sprintf('param.myPercentage{%d} = ceil(csize/2);', p));
            end
        case 3
            %% Clustering algorithm: Fiedler's eigenvector
            % extract eigenvectors
            Di = spdiags(1./sqrt(d'),0,N,N);
            [V,DD] = eigs(param.Laplacian,2,'SA'); % find the 2 smallest eigenvalues and corresponding eigenvectors
            Vv = Di*V; % realign eigenvectors
            v1 = Vv(:,2)/norm(Vv(:,2)); % Fiedler's vector
            % Separate into two communities
            % sweep wrt the ordering identified by v1
            % reorder the adjacency matrix
            [v1s,pos] = sort(v1);
            sortW = W(pos,pos);
            
            % evaluate the conductance measure
            a = sum(triu(sortW));
            b = sum(tril(sortW));
            d = a+b;
            D = sum(d);
            assoc = cumsum(d);
            assoc = min(assoc,D-assoc);
            cut = cumsum(b-a);
            conduct = cut./assoc;
            conduct = conduct(1:end-1);
            % show the conductance measure
            figure('Name','Conductance')
            plot(conduct,'x-')
            grid
            title('conductance')
            
            % identify the minimum -> threshold
            [~,mpos] = min(conduct);
            threshold = mean(v1s(mpos:mpos+1));
            disp(['Minimum conductance: ' num2str(conduct(mpos))]);
            disp(['   Cheeger''s upper bound: ' num2str(sqrt(2*DD(2,2)))]);
            disp(['   # of links: ' num2str(D/2)]);
            disp(['   Cut value: ' num2str(cut(mpos))]);
            disp(['   Assoc value: ' num2str(assoc(mpos))]);
            disp(['   Community size #1: ' num2str(mpos)]);
            disp(['   Community size #2: ' num2str(N-mpos)]);
            param.mpos = mpos;
            param.tpos{1} = pos(1:param.mpos);
            param.tpos{2} = pos(param.mpos+1:end);
            for p = 1:P
                eval(sprintf('param.myPercentage{%d} = ceil(length(param.tpos{%d})/2);', p,p));
                eval(sprintf('W_comm{%d} = W(param.tpos{%d},param.tpos{%d});',p,p,p));
                % Construct the two subLaplacians
                eval(sprintf('TrainSignal{%d} = TRS(param.tpos{%d},:);', p,p));
                eval(sprintf('TestSignal{%d} = TTS(param.tpos{%d},:);',p,p));
                eval(sprintf('param.L = diag(sum(W_comm{%d},2)) - W_comm{%d};',p,p)); % combinatorial Laplacian
                eval(sprintf('param.Laplacian%d = (diag(sum(W_comm{%d},2)))^(-1/2)*param.L*(diag(sum(W_comm{%d},2)))^(-1/2);',p,p,p)); % normalized Laplacian
                eval(sprintf('[param.eigenMat%d, param.eigenVal%d] = eig(param.Laplacian%d);',p,p,p)); % eigendecomposition of the normalized Laplacian
                eval(sprintf('[param.lambda_sym%d,index_sym%d] = sort(diag(param.eigenVal%d));',p,p,p)); % sort the eigenvalues of the normalized Laplacian in descending order
            end
    end
end