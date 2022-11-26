function [history_avgsol_res,history_sol_res,history_objval_res,Iterdiff,sum_avg_grad] = gradConsensus(vec_Dim, numberNodes,...
                                                      Weight_Row_Stoc, A, B, IniEstimate, C, V, Diameter,...
                                                      rho, alpha, index, Xstar, fstar,Max_Iter)
        

               
        history_sol_res = zeros(1,Max_Iter+1);
        history_avgsol_res = zeros(1,Max_Iter+1);
        history_objval_res = zeros(1,Max_Iter+1);        
        
        X = zeros(vec_Dim,numberNodes,Max_Iter+1);
        Z = zeros(vec_Dim,numberNodes,Max_Iter+1);
        X(:,:,1)= IniEstimate;
        Z(:,:,1)= IniEstimate;
        
        Iterdiff = zeros(Max_Iter,1);
        sum_avg_grad = zeros(Max_Iter,1);

        for i = 2:2:Max_Iter
               grad = gradient( A, B, X(:,:,i-1), V, numberNodes, index, vec_Dim);
               for j = 1:numberNodes
                  X(:,j,i) = X(:,j,i-1) - alpha*grad(:,j);
               end
               [Z(:,:,i),Iterdiff(i)] = Consensus(Weight_Row_Stoc,Diameter,X(:,:,i),rho);
               
               avg_X = mean(Z(:,:,i),2);
               Avg_X = avg_X.*ones(vec_Dim,numberNodes);
               avg_grad = gradient( A, B, Avg_X, V, numberNodes, index, vec_Dim);
               sum_avg_grad(i) = norm(sum(avg_grad,2),2);
               
               var_sol = X(:,:,i) - Xstar;
               var_avgsol = Z(:,:,i) - Xstar;
               var_inisol = IniEstimate - Xstar;
               history_sol_res(i) = norm(var_sol,2)^2/norm(var_inisol,2)^2;
               history_avgsol_res(i) = norm(var_avgsol,2)^2/norm(var_inisol,2)^2;
               history_objval_res(i) = abs(objective(A,B,C,Avg_X,V,numberNodes,index) - fstar);
               
                if history_sol_res(i) < 1e-3
                   break;
                end
                
               X(:,:,i+1) = Z(:,:,i); 
        end                
end 