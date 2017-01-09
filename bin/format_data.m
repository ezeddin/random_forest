
X = 2*ones(435,16);
for i = 1:435
    for j = 1:16
        if strcmp(votes(i,j),'n')
            X(i,j) = -1;
        end
        if strcmp(votes(i,j),'y')
            X(i,j) = 1;
        end
        if strcmp(votes(i,j),'?')
            X(i,j) = 0;
        end
    end
end

y = -ones(435,1);
for i = 1:435
   if strcmp(democrat(i),'democrat')
       y(i) = 0;
   end
   if strcmp(democrat(i),'republican')
       y(i) = 1;
   end
%    if strcmp(cp(i),'C')
%        y(i) = 3;
%    end
%    if strcmp(cp(i),'D')
%        y(i) = 4;
%    end
%    if strcmp(cp(i),'E')
%        y(i) = 5;
%    end
%    if strcmp(cp(i),'F')
%        y(i) = 6;
%    end
%    if strcmp(cp(i),'G')
%        y(i) = 7;
%    end
%    if strcmp(cp(i),'H')
%        y(i) = 8;
%    end
%    if strcmp(cp(i),'I')
%        y(i) = 9;
%    end   end
%    if strcmp(cp(i),'J')
%        y(i) = 10;
%    end
%    if strcmp(cp(i),'K')
%        y(i) = 11;
%    end
%    if strcmp(cp(i),'L')
%        y(i) = 12;
%    end
%    if strcmp(cp(i),'M')
%        y(i) = 13;
%    end
%    if strcmp(cp(i),'N')
%        y(i) = 14;
%    end
%    if strcmp(cp(i),'O')
%        y(i) = 15;
%    end
%    if strcmp(cp(i),'P')
%        y(i) = 16;
%    end
%    if strcmp(cp(i),'Q')
%        y(i) = 17;
%    end
%    if strcmp(cp(i),'R')
%        y(i) = 18;
%    end
%    if strcmp(cp(i),'S')
%        y(i) = 19;
%    end
%    if strcmp(cp(i),'T')
%        y(i) = 20;
%    end
%    if strcmp(cp(i),'U')
%        y(i) = 21;
%    end
%    if strcmp(cp(i),'V')
%        y(i) = 22;
%    end
%    if strcmp(cp(i),'W')
%        y(i) = 23;
%    end
%    if strcmp(cp(i),'X')
%        y(i) = 24;
%    end
%    if strcmp(cp(i),'Y')
%        y(i) = 25;
%    end
%    if strcmp(cp(i),'Z')
%        y(i) = 26;
%    end
end