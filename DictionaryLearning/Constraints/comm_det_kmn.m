function [param, W_comm, TrainSignal, TestSignal, idx] = comm_det_kmn(param, W, TRS, TTS, clustN)
    %%%%%%%%%%% BE CAREFUL: this method is different from the one in
    %%%%%%%%%%% dictionary leraning! %%%%%%%%%%
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
    [V,DD] = eig(param.Laplacian); % find the eigenvalues and corresponding eigenvectors
    
    y = 2;
    switch y
        case 1 % Fiedler's eigenvector
            v1 = V(:,2)/norm(V(:,2)); % Fiedler's vector
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
            W1 = W(pos(1:mpos),pos(1:mpos));
            W2 = W(pos(mpos+1:N),pos(mpos+1:N));
            param.pos = pos;
            param.mpos = mpos;
            param.pos1 = param.pos(1:param.mpos);
            param.pos2 = param.pos(param.mpos+1:end);
            % Construct the two subLaplacians
            for p = 1:2
                eval(['TrainSignal{',num2str(p),'} = TRS(param.pos',num2str(p),',:);']);
                eval(['TestSignal{',num2str(p),'} = TTS(param.pos',num2str(p),',:);']);
                eval(['param.L',num2str(p),' = diag(sum(W',num2str(p),',2)) - W',num2str(p),';']); % combinatorial Laplacian
                eval(['param.Laplacian',num2str(p),' = (diag(sum(W',num2str(p),',2)))^(-1/2)*param.L',num2str(p),'*(diag(sum(W',num2str(p),',2)))^(-1/2);']); % normalized Laplacian
                eval(['[param.eigenMat',num2str(p),', param.eigenVal',num2str(p),'] = eig(param.Laplacian',num2str(p),');']); % eigendecomposition of the normalized Laplacian
                eval(['[param.lambda_sym',num2str(p),',index_sym',num2str(p),'] = sort(diag(param.eigenVal',num2str(p),'));']); % sort the eigenvalues of the normalized Laplacian in descending order
            end
        case 2 % k-means approach
            % Max eigengap
            [DD, pos] = sort(diag(DD));
            gap = DD(1:end-1) - DD(2:end);
            [~, gapPos] = max(gap); % The maximum gap position identifies the number of
            % eigenvalues/vectors to be taken to span the
            % subspace and so the number of centroids
            
            subspace_dim = gapPos - 1; % number of eigenvalues I take to project the signal
            W_comm = cell(clustN,1);
            TrainSignal = cell(clustN,1);
            TestSignal = cell(clustN,1);
            param.tpos = cell(clustN,1); % The cell matrix containing in each cell the list
            % of nodes belonging to cluster i
            
            % Take the subspace_dim smallest eigenvalues and corresponding eigenvectors
            X = V(:,pos(2:subspace_dim+1));
            % Normalize the rows of X
            for i = 1:N
                X(i,:) = X(i,:)./norm(X(i,:));
            end
            
            % Clustering algorithm: k-means approach
            idx = kmeans(X,clustN);
            
            for p = 1:clustN
                tmp = (idx == p);
                param.tpos{p} = find(tmp);
                csize = sum(tmp);
                W_comm{p} = W(param.tpos{p},param.tpos{p});
                TrainSignal{p} = TRS(param.tpos{p},:);
                TestSignal{p} = TTS(param.tpos{p},:);
                disp(['Community size #',num2str(p),': ', num2str(csize)]);
                
                exp1 = sprintf('param.L = diag(sum(W_comm{%d},2)) - W_comm{%d};',p,p);
                exp2 = sprintf('param.Laplacian%d = (diag(sum(W_comm{%d},2)))^(-1/2)*param.L*(diag(sum(W_comm{%d},2)))^(-1/2);', p,p,p);
                exp3 = sprintf('[param.eigenMat%d, param.eigenVal%d] = eig(param.Laplacian%d);', p,p,p);
                exp4 = sprintf('[param.lambda_sym%d, index_sym%d] = sort(diag(param.eigenVal%d));', p,p,p);
                eval(exp1); eval(exp2); eval(exp3); eval(exp4);
                eval(sprintf('param.myPercentage{%d} = ceil(csize/2);', p));
            end
            
    end
end