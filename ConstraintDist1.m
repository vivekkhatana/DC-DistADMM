function [sol_res, objval_res, comm, constraint_res, ftime, eq_constraint_res] = ConstraintDist1(vec_Dim, numberNodes,...
                                                      Laplacian_Mat, A, B, IniEstimate, C, H, h, index,...
                                                      Xstar, fstar, Max_Iter, mu, acc)
        
        % Algorithm from:
        % Liu, Yang, Hong, Constrained Consensus Algorithms With Fixed Step Size for Distributed Convex Optimization Over Multiagent Networks, TAC, 2017
               
        sol_res = []; 
        objval_res = [];       
        constraint_res = [];
        eq_constraint_res = [];
        tX = [];
        tY = []; 
        tW =[];
        tX_hat = [];
        tX(:,:,1) = IniEstimate;
        [t1,t2] = size(H);
        tY(:,:,1) = zeros(t1,numberNodes);
        tW(:,:,1) = IniEstimate;
        tX_hat(:,:,1) = IniEstimate;
        comm = [];
        
        itime = cputime;
 
        for i = 1:1:Max_Iter
               tic; 
               if index == 3
                 
                     alpha = 0.01;
                     grad = gradient( A, B, tX(:,:,i), numberNodes, mu, index);
                     subgrad = subgradient(tX(:,:,i), mu);
                     
%                      for jj = 1:numberNodes
%                          zz = tX(:,jj,i);
%                          for ij = 1:vec_Dim
%                            if zz(ij) > 0
%                                subgrad(ij,jj) = 1*10;
%                            else
%                                subgrad(ij,jj) = -1*10;
%                            end
%                          end
%                      end

                     dummy = [];
                     for j = 1:numberNodes
                         dummy(:,j) = H'*(tY(:,j,i) + H*tX(:,j,i) - h);
                     end
                     
                     X =  tX(:,:,i) - alpha*( grad + subgrad + dummy + tW(:,:,i) + tX(:,:,i)*Laplacian_Mat ); 

                     % projection
                     for j = 1:numberNodes
                         if norm(X(:,j),2) <= 1
                                   X(:,j) = X(:,j);
                               else
                                   X(:,j) = X(:,j)/norm(X(:,j),2);
                         end 
                     end

                     tX(:,:,i+1) = X;
                     
                     for j = 1:numberNodes
                         tY(:,j,i+1) = tY(:,j,i) + H*tX(:,j,i+1) - h;
                     end
                     tW(:,:,i+1) = tW(:,:,i) + tX(:,:,i+1)*Laplacian_Mat;

                     comm(i) = 1;
               
               end

               tX_hat(:,:,i+1) = 0.5*(tX_hat(:,:,i) + tX(:,:,i+1));
               
               var_sol = tX(:,:,i) - Xstar;
               var_inisol = IniEstimate - Xstar;

               sol_res(i) = (1/i^0.1)*norm(var_sol,'fro')^2/norm(var_inisol,'fro')^2;
               objval_res(i) = objective(A,B,C,tX_hat(:,:,i),numberNodes,mu,index);

               constraint_res(i) =  norm( tX_hat(:,:,i+1) - mean(tX(:,:,i+1),2),'fro');
               
               eq_cons = 0;
                   for j = 1:numberNodes
                     eq_cons = eq_cons + norm( H*tX_hat(:,j,i+1) -  h,'fro');
                   end
               eq_constraint_res(i) = (1/i^0.1)*eq_cons;
               
              ftime(i) = toc; 
              
              if acc ~= 0
                  if sol_res(i) <= acc
                       break;
                    else 
                        comm = 0;
                  end
              end
                 
        end                
end 