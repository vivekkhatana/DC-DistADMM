function [sol_res, objval_res, comm,ftime] = DGD(numberNodes,...
                                                      Weight_Doub_Stoc, A, B, IniEstimate, C, ...
                                                      index, Xstar, fstar, Max_Iter, mu,acc)

        
        
        sol_res = [];
        objval_res = [];  
        ftime = [];
        
        X = [];
        X_hat = [];
        X(:,:,1)= IniEstimate;
        X_hat(:,:,1) = IniEstimate;              

        alpha = 0.0005;
        
        itime = cputime;
        
        for i = 1:1:Max_Iter
               grad = gradient( A, B, X(:,:,i), numberNodes, mu, index);
               subgrad = subgradient( X(:,:,i), 3);
               var = X(:,:,i)*Weight_Doub_Stoc;
               X(:,:,i+1) =  var - alpha*(grad + subgrad);
               
               X_hat(:,:,i+1) = 0.5*(X_hat(:,:,i) + X(:,:,i+1));                              
               var_sol = X(:,:,i) - Xstar;
               var_inisol = IniEstimate - Xstar;
               sol_res(i) = (1/i^2)*norm(var_sol,'fro')^2/norm(var_inisol,'fro')^2;
               objval_res(i) = objective(A,B,C,X_hat(:,:,i+1),numberNodes, mu, index);
               
               ftime(i) = cputime - itime;
               
              if acc ~= 0
                  if sol_res(i) <= acc
                       comm = i;
                       break;
                    else 
                        comm = 0;
                  end
              else
                comm = 0; 
              end
        end
                                        
end

 