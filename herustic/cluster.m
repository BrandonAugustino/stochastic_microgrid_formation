function cluster(Nn,genloc)
tic 
load('tomatlab.dat');
Gcluster = graph(tomatlab(:,1),tomatlab(:,2));
Dmatrix = distances(Gcluster,genloc);
nDG = length(genloc);



Clust = cell(nDG,1);

for i=1:Nn
    nwclust_id = find(Dmatrix(:,i)==min(Dmatrix(:,i)),1);
    Clust{nwclust_id} =  [Clust{nwclust_id},i];  
end
delete 'topython.dat'
for k=1:nDG
    dlmwrite('topython.dat',unique(Clust{k,1}(1,:)),'-append');
end
toc 