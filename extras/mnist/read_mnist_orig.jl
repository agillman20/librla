
### http://yann.lecun.com/exdb/mnist/

### wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
### wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
### wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
### wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

### git clone https://github.com/fgnt/mnist
### rsync -apv mnist/*.gz .
### gunzip *.gz


include("compat.jl")


function read_mnist()

f = open("train-labels-idx1-ubyte","r")

magic = ntoh(read(f,Int32))
count = ntoh(read(f,Int32))

train_labels = Array{UInt8}(undef,count)
read!(f,train_labels)

println("magic=", magic)
println("count=", count)
##println("train_labels=", train_labels)

close(f)


f = open("train-images-idx3-ubyte","r")

magic = ntoh(read(f,Int32))
count = ntoh(read(f,Int32))
nrows = ntoh(read(f,Int32))
ncols = ntoh(read(f,Int32))

train_images = Array{UInt8}(undef,nrows,ncols,count)
read!(f,train_images)

##train_images = permutedims(train_images, [2 1 3]);

println("magic=", magic)
println("count=", count)
println("nrows=", nrows)
println("ncols=", ncols)
##println("train_images=", train_images)

close(f)


f = open("t10k-labels-idx1-ubyte","r")

magic = ntoh(read(f,Int32))
count = ntoh(read(f,Int32))

test_labels = Array{UInt8}(undef,count)
read!(f,test_labels)

println("magic=", magic)
println("count=", count)
##println("test_labels=", test_labels)

close(f)


f = open("t10k-images-idx3-ubyte","r")

magic = ntoh(read(f,Int32))
count = ntoh(read(f,Int32))
nrows = ntoh(read(f,Int32))
ncols = ntoh(read(f,Int32))

test_images = Array{UInt8}(undef,nrows,ncols,count)
read!(f,test_images)

##test_images = permutedims(test_images, [2 1 3]);

println("magic=", magic)
println("count=", count)
println("nrows=", nrows)
println("ncols=", ncols)
##println("test_images=", test_images)

close(f)


train_labels[find(train_labels .== 0 )] .= 10
test_labels[find(test_labels .== 0 )] .= 10

return train_labels, train_images, test_labels, test_images

end


train_labels, train_images, test_labels, test_images = read_mnist()


