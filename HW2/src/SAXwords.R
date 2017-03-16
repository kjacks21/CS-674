library(TSMining)
library(doMC)
registerDoMC(4)

# number of datasets
for (i in 1:5){
  
  test_path = sprintf("/home/kyle/Documents/CS-674/HW2/hw2_datasets/hw2_datasets/dataset%d/test.txt", i)
  train_path = sprintf("/home/kyle/Documents/CS-674/HW2/hw2_datasets/hw2_datasets/dataset%d/train.txt", i)
  test = read.table(test_path)
  train = read.table(train_path)
  
  test_label = test[,1]
  test = test[,-1]
  
  train_label = train[,1]
  train = train[,-1]
  
  # write sax for train
  train_datalist = list()
  for (n in 1:dim(train)[1]) {
    saxWord = Func.SAX(x=as.numeric(train[n,]), w=dim(train)[2], a=20, eps=.01, norm=FALSE)
    sax.list = as.list(saxWord)
    train_datalist[[n]] = saxWord # add it to your list
    print(sprintf("train %d complete", n))
  }
  big_data = do.call(rbind, train_datalist)
  sax_train_path = sprintf("/home/kyle/Documents/CS-674/HW2/sax_data/dataset%d/train_sax2.txt", i)
  write.table(big_data,sax_train_path,sep="\t",row.names=FALSE,col.names=FALSE)
  
  # write sax for test
  test_datalist = list()
  for (n in 1:dim(test)[1]) {
    saxWord = Func.SAX(x=as.numeric(test[n,]), w=dim(test)[2], a=20, eps=.01, norm=FALSE)
    sax.list = as.list(saxWord)
    test_datalist[[n]] = saxWord # add it to your list
    print(sprintf("test %d complete", n))
  }
  big_data = do.call(rbind, test_datalist)
  sax_test_path = sprintf("/home/kyle/Documents/CS-674/HW2/sax_data/dataset%d/test_sax2.txt", i)
  write.table(big_data,sax_test_path,sep="\t",row.names=FALSE,col.names=FALSE)
}

# motif discovery
test_path = "/home/kyle/Documents/CS-674/HW2/hw2_datasets/hw2_datasets/dataset1/test_X.txt"
train_path = "/home/kyle/Documents/CS-674/HW2/hw2_datasets/hw2_datasets/dataset1/train_X.txt"
test = read.table(test_path)
train = read.table(train_path)

result <- Func.motif(ts = as.numeric(test[1,]), global.norm = FALSE, local.norm = FALSE,
                    window.size = 10, overlap = 0, w = 5, a = 3, mask.size = 3, eps = .01)

#Check the number of motifs discovered
my_list = c()
for (i in result$Motif.SAX){
  d = apply(i,1,paste,collapse="")
  df = data.frame(as.list(d))
  vals = as.character(unlist(df[1,]))
  my_list = c(vals, my_list)
}



