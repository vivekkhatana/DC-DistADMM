function [sol_res, objval_res, comm, constraint_res, ftime, eq_constraint_res] = ConstraintDist2(vec_Dim, numberNodes,...
                                                      Weight_Mat, A, B, IniEstimate, C, H, h, index,...
                                                      Xstar, fstar, Max_Iter, mu, acc)
        
        % Algorithm from:
        % Zhu, Martinez, On Distributed Convex Optimization Under Inequality and Equality Constraints, TAC, 2012
               
        sol_res = []; 
        objval_res = [];       
        constraint_res = [];
        eq_constraint_res = [];
        objval_res(1) = 0;
        sol_res(1) = 1;
        constraint_res(1) = 100;
        eq_constraint_res(1) = 1000;
        X = [];
        Y = []; 
        Lam =[];
        X_hat = [];
        X(:,:,1) = IniEstimate;
        Y(:,:,1) = IniEstimate;
        [t1,t2] = size(H);
        Lam(:,:,1)= zeros(t1,numberNodes);
        X(:,:,2) = IniEstimate;
        Y(:,:,2) = IniEstimate;
        Lam(:,:,2)= zeros(t1,numberNodes);
        X_hat(:,:,1) = IniEstimate;
        X_hat(:,:,2) = IniEstimate;
        comm = [];
        
        itime = cputime;
 
        for i = 2:1:Max_Iter
               tic; 
               if index == 3
                 
                     alpha = 1/(i+1)^1.2;
                     vX = X(:,:,i)*Weight_Mat;
                     vY = Y(:,:,i)*Weight_Mat;
                     vLam = Lam(:,:,i)*Weight_Mat;                    
                     
                     grad = gradient( A, B, vX, numberNodes, mu, index);
                     subgrad = subgradient(vX, mu);
                     
%                      for jj = 1:numberNodes
%                          zz = vX(:,jj);
%                          for ij = 1:vec_Dim
%                            if zz(ij) > 0
%                                subgrad(ij,jj) = 1*10;
%                            else
%                                subgrad(ij,jj) = -1*10;
%                            end
%                          end
%                      end                  
                     
                     tH = [];
                     for j = 1:numberNodes
                         tH(:,j) = vLam(:,j)'*H;
                     end
                     
                     grad2 = subgradient(tH, 1);
%                      for j = 1:numberNodes
%                          for  l = 1:vec_Dim
%                              if (tH(l,j) >= 0)
%                                  grad2(l,j) = 1;
%                              elseif (tH(l,j) < 0)
%                                  grad2(l,j) = -1;
%                              end
%                          end
%                      end
                     
                     tX =  X(:,:,i) - alpha*( grad + subgrad + grad2 ); 

                     % projection
                     for j = 1:numberNodes
                         if norm(tX(:,j),2) <= 1
                                   tX(:,j) = tX(:,j);
                               else
                                   tX(:,j) = tX(:,j)/norm(tX(:,j),2);
                         end 
                     end 

                     X(:,:,i+1) = tX;
                     
                     Y(:,:,i+1) = vY + numberNodes*( objective(A,B,C,X(:,:,i),numberNodes,mu,index) ...
                                                     - objective(A,B,C,X(:,:,i-1),numberNodes,mu,index) );

                     for j = 1:numberNodes
                         Lam(:,j,i+1) = vLam(:,j) + alpha*(H*vX(:,j) - h);
                     end

                     comm(i) = 1;
               
               end

               var_sol = X(:,:,i) - Xstar;
               var_inisol = IniEstimate - Xstar;
               X_hat(:,:,i+1) = 0.5*(X_hat(:,:,i) + X(:,:,i+1));

               sol_res(i) = (1/i^0.3)*norm(var_sol,'fro')^2/norm(var_inisol,'fro')^2;
               objval_res(i) = (1/i^0.3)*objective(A,B,C,X_hat(:,:,i),numberNodes,mu,index);

               constraint_res(i) =  (1/i^0.3)*norm( X_hat(:,:,i+1) - mean(X(:,:,i+1),2),'fro');
               
               eq_cons = 0;
                   for j = 1:numberNodes
                     eq_cons = eq_cons + norm( H*X_hat(:,j,i+1) -  h,'fro');
                   end
               eq_constraint_res(i) = (1/i^0.3)*eq_cons;
               
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