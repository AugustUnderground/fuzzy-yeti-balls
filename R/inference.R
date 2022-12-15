library("torch")

model_path <- "../../prehsept/models/xh018/nmos-20221129-073825/trace.pt"
model_path <- "../models/trace.pt"
model <- jit_load(model_path)
x <- torch_reshape(torch_tensor(c(10, 10^7, 0.666, 0.0)), list(1, 4))
y <- model(x)
