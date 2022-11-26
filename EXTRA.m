function [sol_res, objval_res, comm, ftime] = EXTRA(numberNodes,...
                                                      Weight_Doub_Stoc, A, B, IniEstimate, C, ...
                                                      index, Xstar, fstar,Max_Iter,mu,acc)

       
        
        sol_res = [];
        sol_res(1) = 1;
        objval_res = [];   
        ftime = [];
        
        X = [];
        X_hat = [];
        X(:,:,1)= IniEstimate;
        X_hat(:,:,2) = IniEstimate;
%         alpha = 0.0005276;
        alpha = 0.0009;
        grad = gradient( A, B, X(:,:,1), numberNodes, mu, index);
        subgrad = subgradient( X(:,:,1), 3);
        
        X(:,:,2)= X(:,:,1)*Weight_Doub_Stoc - alpha*(grad + subgrad);            
        Weight_Doub_Stoc_tilde = 0.5*( eye(length(Weight_Doub_Stoc)) + Weight_Doub_Stoc );        
        
        itime = cputime;
        
        for i = 2:1:Max_Iter
               var = gradient(A, B, X(:,:,i), numberNodes, mu, index) + subgradient( X(:,:,i), 3)...
                   - gradient(A, B, X(:,:,i-1), numberNodes, mu, index) - subgradient( X(:,:,i-1), 3);
               
               X(:,:,i+1) = X(:,:,i)*2*Weight_Doub_Stoc_tilde - X(:,:,i-1)*Weight_Doub_Stoc_tilde - alpha*var;
               
               X_hat(:,:,i+1) = 0.5*(X_hat(:,:,i) + X(:,:,i+1));             
               var_sol = X(:,:,i) - Xstar;            
               var_inisol = IniEstimate - Xstar;
               sol_res(i) = (1/i^2)*norm(var_sol,'fro')^2/norm(var_inisol,'fro')^2;
               objval_res(i) = objective(A,B,C,X_hat(:,:,i+1),numberNodes,mu,index);
              
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

 