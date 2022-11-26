%get_neighbors

% P is the adjacency matrix and i is the node for which neighbors are being
% queried
function neighbors = get_neighbors(P,i)
    neighbors = [];
    for j = 1:length(P)
        if P(i,j) ~= 0
           neighbors = [neighbors j];
        end
    end

end