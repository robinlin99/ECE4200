[X,Y] = meshgrid(1:0.5:10,1:20);
Z1 = 0.5*X + 0.5*Y;
Z2 = (X.*Y).^0.5;
figure
colormap winter;
surf(X,Y,Z1)
hold on
colormap summer;
surf(X,Y,Z2)
xlabel('X');
ylabel('Y');
zlabel('Z');

