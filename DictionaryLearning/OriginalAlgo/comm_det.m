function [param, W1, W2, TrainSignal, TestSignal] = comm_det(param, W, TRS, TTS)
    % extract eigenvectors
    N = param.N;
    [V,DD] = eigs(param.Laplacian,3,'SA'); % find the 3 smallest eigenvalues and corresponding eigenvectors
    v1 = V(:,2)/norm(V(:,2)); % Fiedler's vector

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
end