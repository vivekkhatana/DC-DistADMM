function grad = gradient( A, B, X, Num_Agents, mu, index)

 grad = [];
 
  if index == 1
      for j = 1:Num_Agents
        grad(:,j) = A{j}*X(:,j) + B(:,j);
      end      
      
  elseif index == 2
      for j = 1:Num_Agents
        grad(:,j) = A{j}'*(A{j}*X(:,j) - B(:,j));
      end 
      
  elseif index == 3 % gradient here is only for the differentiable part, subgradient of \ell_1 norm to be calculated separately
        for j = 1:Num_Agents
           [sample_size,vec_dim] = size(A{j});
           z = X(:,j);
           for k = 1:sample_size
               var = 1 + exp( -B(k,j)*( A{j}(k,:)*z ) );
               var1 = exp( -B(k,j)*( A{j}(k,:)*z ) )*(-B(k,j)*( A{j}(k,:))); 
%                var2 = exp( -B(k,j)*( A{j}(k,:)*z(2:end) + z(1) ) )*(-B(k,j)); 
               innergrad(:,k) = (1/var)*var1';
           end
%            for i = 1:vec_dim
%                if z(i) > 0
%                    subgrad(i,j) = 1;
%                else
%                    subgrad(i,j) = -1;
%                end
%            end
           grad(:,j) = sum(innergrad,2);% + mu*subgrad(:,j);
        end 
       
  elseif index == 4 % gradient here is only for the differentiable part, subgradient of \ell_1 norm to be calculated separately
      for j = 1:Num_Agents
           if (norm(A{j}*X(:,j) - B(:,j),2) < 1)
               grad(:,j) = A{j}'*(A{j}*X(:,j) - B(:,j));
           else
               [sample_size,vec_dim] = size(A{j});
               M = A{j};
               mb = B(:,j);
               gg = [];
               for jj = 1:vec_dim
                  gg(jj) = 2*(M(jj,:)*X(:,j) - mb(jj))*M(jj,jj);
               end
               grad(:,j) = (2/norm(A{j}*X(:,j) - B(:,j),2))*gg;
           end
           
      end
 
  end
  
end