import torch as pt

mdl = pt.jit.load('../models/trace.pt').train()
xs = pt.randn(10,6)
mdl(xs)

pt.onnx.export(mdl, xs, "../models/trace.onnx")

