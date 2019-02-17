function S = link_prediction(W_bd,flag)
    N = length(W_bd);
    switch flag
        case 1
            % Common neighbour technique
            S = W_bd*W_bd;
        case 2
            % Adamic Adar technique
            for i = 1:N
                for j = 1:N
                    cn(j,:) = W_bd(i,:).*W_bd(j,:);
                    k = find(cn(j,:)); %common neighbours to both i and j
                    N_kaa = 0;
                    for r = k
                        if find(W_bd(r,:)) > 1
                            N_kaa = N_kaa + (1/log(sum(W_bd(r,:))));
                        end
                    end
                    S(i,j) = N_kaa;
                end
            end
        case 3
            % Resource Allocation technique
            for i = 1:N
                for j = 1:N
                    cn(j,:) = W_bd(i,:).*W_bd(j,:);
                    k = find(cn(j,:)); %common neighbours to both i and j
                    N_kra = 0;
                    for r = k
                        if ~isempty(r)
                            N_kra = N_kra + (1/sum(W_bd(r,:))); % The sum goes to inf because there are very many common neighbours with a low degree
                        end
                    end
                    S(i,j) = N_kra;
                end
            end
    end
end