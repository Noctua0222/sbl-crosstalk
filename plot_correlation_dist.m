newdata_n=readtable('correlated_n.csv');
newdata_p=readtable('correlated_p.csv');

%\\\\\\\\\\\\\\\\\\\\\\\\Histogram////////////////////////////////
% figure(1)
% hold on
% histogram(dataout{:,7})
% histogram(newdata{:,7})
% hold off
% 
% title('Distribution of Degree')
% legend('Original','Genes Display Crosstalk')
% 
% figure(2)
% hold on
% histogram(dataout{:,8})
% histogram(newdata{:,8})
% hold off
% 
% title('Distribution of Betweenness')
% legend('Original','Genes Display Crosstalk')
% 
% figure(3)
% hold on
% histogram(dataout{:,6})
% histogram(newdata{:,6})
% hold off
% 
% title('Distribution of Closeness')
% legend('Original','Genes Display Crosstalk')
% 
% figure
% hold on
% histogram(dataout{:,9})
% histogram(newdata{:,9})
% hold off
% 
% title('Distribution of Pagerank')
% legend('Original','Genes Display Crosstalk')

%\\\\\\\\\\\\\\\\\\\\\\\\\\\\Boxplot//////////////////////////////////////

g=[repmat({'original'},size(dataout,1),1); repmat({'correlated_n'},size(newdata_n,1),1);repmat({'correlated_p'},size(newdata_p,1),1)];

f=figure;
subplot(2,2,1);
boxplot([dataout{:, 6};newdata_n{:,6};newdata_p{:,6}],g,'Labels',{'Overall distribution','Negatively_Correlated','Positively_Correlated'});
title('Distribution of Closeness')
subplot(2,2,2);
boxplot([dataout{:, 7};newdata_n{:,7};newdata_p{:,7}],g,'Labels',{'Overall distribution','Negatively_Correlated','Positively_Correlated'});
title('Distribution of Degree')
subplot(2,2,3);
boxplot(log([dataout{:, 8};newdata_n{:,8};newdata_p{:,8}]),g,'Labels',{'Overall distribution','Negatively_Correlated','Positively_Correlated'});
title('Distribution of Betweenness (on log scale)')
subplot(2,2,4);
boxplot([dataout{:, 9};newdata_n{:,9};newdata_p{:,9}],g,'Labels',{'Overall distribution','Negatively_Correlated','Positively_Correlated'});
title('Distribution of Pagerank')
f.Position=[0,0,1500,1000];
saveas(f,'Comparison of distribution with Threshold=0.7.png');
