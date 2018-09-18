function FigureClassifierDependence
% Plots two figures which demonstrate that feature selection should be
% classifier-dependent. Some features may be useful for one classifier and
% useless for another. The example here shows a pair of features which are
% suitable for LDC and useless for 1NN according to the Leave-One-Out (LOO)
% criterion, and another pair which is completely the opposite.
%
% (c) Lucy Kuncheva 18/09/2018

% Feature pair #1 ---------------------------------------------------------
a = 1:2:9; b = 2:2:10; d = [a' b';b' a']; % data
l = [ones(5,1);ones(5,1)*2]; % labels
set_up_figure(d,l)
[e1nn1, eldc1] = calculate_error(d,l); % needs Statistics Toolbox
fprintf('Ex 1. 1NN %.4f LCD %.4f\n', e1nn1, eldc1)

% Feature pair #2 ---------------------------------------------------------
N = 24;
t = linspace(0,2*pi,N+1);
tt = t(1:end-1);
l = repmat([1 1 1 2 2 2]',N/6,1); % labels
x(l==1) = sin(tt(l==1))*4.5+5;
x(l==2) = sin(tt(l==2))*4+5;
y(l==1) = cos(tt(l==1))*4.5+5;
y(l==2) = cos(tt(l==2))*4+5;
d = [x(:) y(:)]; % data
set_up_figure(d,l)
[e1nn2, eldc2] = calculate_error(d,l);
fprintf('Ex 2. 1NN %.4f LCD %.4f\n', e1nn2, eldc2)

end

function set_up_figure(d,l)

figure, hold on, grid on
set(gca,'FontName','Times','XTick',0:2:10,'YTick',0:2:10,...
    'XTickLabel','','YTickLabel','','FontSize',30)
axis([0 10 0 10])
axis square
plot(d(l == 1,1),d(l == 1,2),'bo','markersize',12,'markerfacecolor','b')
plot(d(l == 2,1),d(l == 2,2),'r^','markersize',12,'markerfacecolor','r')
xlabel('{\it x}_1')
ylabel('{\it x}_2')
yl = get(gca,'YLabel');
set(yl,'Position',[-1.2 9.2],'Rotation',0)
xl = get(gca,'XLabel');
set(xl,'Position',[11 1],'Rotation',0)
end

function [e1nn, eldc] = calculate_error(d,l)
c = fitcknn(d,l,'Leaveout','on');
e1nn = mean(l ~= kfoldPredict(c));

c = fitcdiscr(d,l,'Leaveout','on');
eldc = mean(l ~= kfoldPredict(c));
end
