function spectrum = spectral_rep(eigenvalues_vector)
    % Obtaining a spectral representation of the signal
    % Frequencies = eigenvalues of the graph laplacian
% % %     eigenvalues_vector = eigenvalues*ones(length(eigenvalues),1);
    eigenvalues_vector = sort(eigenvalues_vector);
    i = 1;
    j = 1;
    while i <= length(eigenvalues_vector)
        counter = 1;
        while i < length(eigenvalues_vector) && eigenvalues_vector(i) == eigenvalues_vector(i+1)
            counter = counter + 1;
            i = i + 1;
        end
        recurrences(j) = counter;
        frequencies(j) = eigenvalues_vector(i);
        i = i + 1;        
        j = j+1;
    end
    
    figure('Name', 'Spectrum of the signal')
    plot(frequencies,recurrences);
    
    spectrum = [frequencies; recurrences];    
end