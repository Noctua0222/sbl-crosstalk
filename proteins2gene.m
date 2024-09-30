%correlation_n/p.csv: positively or negatively correlated protein from
%cancer paper S6I
%map.csv: mapping source downloaded from uniprot.com
%carom.csv: gene network characteristic from carom paper S1S

%read data
datain=readtable('correlation_n.csv');%change to _n for negatively correlated
map=readtable('map.csv');
dataout=readtable('carom.csv');

%set correlation threshold
threshold=0.7;

%find namespaces that need to be mapped
proteins=string(datain{:,1});
proteins=proteins(find(abs(datain{:,12})>threshold));
tg1=string(map{:,1});
tg2=string(dataout{:,1});

%map the protein names to gene names (NCBI refseq to Gene ID)
names=[];
for i=1:size(proteins,1)
    ind=find(strcmp(proteins(i),tg1));
    for j=1:sum(~ismissing(map{ind,6}))
    num=find(~ismissing(map{ind,6}));
    names=[names split(string(map{ind(num(j)),6}))'];
    end
end

%Only keep unique gene entries
names=unique(names.');

%map the genes to entries in carom paper, extract to a new csv file
for i=1:size(names,1)
    ind=find(strcmp(names(i),tg2));
    if i==1
        writetable(dataout(ind,:),'correlated_n.csv','WriteMode','overwrite')
    else
        writetable(dataout(ind,:),'correlated_n.csv','WriteMode','append')
    end
end



