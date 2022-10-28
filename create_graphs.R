library(igraph)
library(MASS)

sbm <- function(filename){
    edges<-as.matrix(read.table(filename,header=FALSE))
    sbm <- graph_from_edgelist(edges, directed = TRUE)
    return(sbm)
}

create.synthetic <- function(){
    small500 <- watts.strogatz.game(1, 500, 5, 0.05)
    small1000 <- watts.strogatz.game(1, 1000, 5, 0.05)
    small5000 <- watts.strogatz.game(1, 5000, 5, 0.05)

    fire500 <- forest.fire.game(n=500, fw.prob=0.32, bw.factor=0.33/0.32, directed=FALSE)
    fire1000 <- forest.fire.game(n=1000, fw.prob=0.37, bw.factor=0.33/0.37, directed=FALSE)
    fire5000 <- forest.fire.game(n=5000, fw.prob=0.37, bw.factor=0.35/0.37, directed=FALSE)

    sbm500 <- sbm(paste(getwd(),"/synthetic/sbm500/network.dat"))
    sbm1000 <- sbm(paste(getwd(),"/synthetic/sbm1000/network.dat"))
    sbm5000 <- sbm(paste(getwd(),"/synthetic/sbm5000/network.dat"))

    write.matrix(get.adjacency(small500), paste(getwd(),"synthetic/small500.csv"))
    write.matrix(get.adjacency(small1000), paste(getwd(),"synthetic/small1000.csv"))
    write.matrix(get.adjacency(small5000), paste(getwd(),"synthetic/small5000.csv"))

    write.matrix(get.adjacency(fire500), paste(getwd(),"synthetic/fire500.csv"))
    write.matrix(get.adjacency(fire1000), paste(getwd(),"synthetic/fire1000.csv"))
    write.matrix(get.adjacency(fire5000), paste(getwd(),"synthetic/fire5000.csv"))

    write.matrix(get.adjacency(sbm500), paste(getwd(),"synthetic/sbm500.csv"))
    write.matrix(get.adjacency(sbm1000), paste(getwd(),"synthetic/sbm1000.csv"))
    write.matrix(get.adjacency(sbm5000), paste(getwd(),"synthetic/sbm5000.csv"))
}

create.synthetic()