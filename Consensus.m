function [AvgVal,iteration] = Consensus(Weight_Matrix,D,X0,rho)
  
                  X = X0;
%                   [row, col] = size(X0);
%                   Y = ones(row, col);
                  z =[];
                  w =[];    
                  z_data = [];        
                  diam= D;
                  count= 0;
                  diff = 100;
                  iter = 0;

        while diff > rho
    
             X = X*Weight_Matrix;
%              Y = Y*Weight_Matrix;
             iter = iter+1;
%              R = bsxfun(@rdivide, X, Y);
             z = [z; max(X)];
%              z = [z; max(R)];
%              w = [w; min(R)]; 
%              z_data = [z_data; R];
             z_data = [z_data; X];
             count=count+1;
           
          if count == diam
                MNP = min(z);
                MXP = max(z);
                 diff = abs(norm(MXP)-norm(MNP));
                z = [];
%                 w = [];
                count=0;
          end

        end  
AvgVal = X;
% AvgVal = R;
iteration = iter;
