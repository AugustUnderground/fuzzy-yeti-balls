modelfile = "../models/trace.pt"
net = importNetworkFromPyTorch(modelfile)

ys_ = predict(net, xs)
