matrix = [5 NaN 1 2 3 4; 5 4 2 2 2 5; 1 2 4 NaN 4 3; NaN 2 3 5 NaN NaN; NaN 3 5 4 5 1];

% movie comparison
for i=1:5
    for j=i+1:6
        r1 = [];
        r2 = [];
        notnan1 = [];
        notnan2 = [];
        for k=1:5
            if ~isnan(matrix(k,i)) && ~isnan(matrix(k,j))
                r1 = [r1 matrix(k,i)];
                r2 = [r2 matrix(k,j)];
            end
        end
        for k=1:5
            if ~isnan(matrix(k,i)) 
                notnan1 = [notnan1 matrix(k,i)];
            end
            if ~isnan(matrix(k,j)) 
                notnan2 = [notnan2 matrix(k,j)];
            end
        end
        s1 = size(r1);
        s2 = size(r2);
        avg1 = mean(notnan1)*ones(1,s1(1,2));
        avg2 = mean(notnan2)*ones(1,s2(1,2));
        
        eucl = 1/norm(r1-r2);
        pearson = dot(r1-avg1,r2-avg2)/(norm(r1-avg1)*norm(r2-avg2));
        disp("i:")
        disp(i);
        disp("Avg 1");
        disp(avg1);
        disp("j:")
        disp(j)
        disp("Avg 2");
        disp(avg2);
        disp("Euclid")
        disp(eucl)
        disp("Pearson")
        disp(pearson)
        disp("---------")
        
    end
end