function X = generate_sparsity(n_ker,t0,m,l)
    X = zeros(n_ker*m,l);
    for i = 1:l
        indexes = randperm(m*n_ker);
        indexes = indexes(1:t0);
        for j = 1:length(indexes)
            X(indexes(j),i) = rand;
        end
    end
    
end