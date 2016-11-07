
Look at main.lua, you should be able to get some ideal

## Dataset

local loader = require "iris\_loader"

dataset = loader.load_data()
x = dataset.inputs
y = dataset.targets

x is 150x4 input data
with 150 sample, 4 column each

y is labels with 4 class 1, 2, 3, 4

## Model
look in model folder
### SimpleSeqModel

x -> nn.Linear(4,4) -> nn.LogSoftMax() -> y



