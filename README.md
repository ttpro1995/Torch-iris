
Look at main.lua, you should be able to get some ideal

## Dataset

local loader = require "iris\_loader" <br>

dataset = loader.load_data() <br>
x = dataset.inputs <br>
y = dataset.targets <br>

x is 150x4 input data <br>
with 150 sample, 4 column each <br>

y is labels with 4 class 1, 2, 3, 4 <br>

## Model 
look in model folder
### SimpleSeqModel

x -> nn.Linear(4,4) -> nn.LogSoftMax() -> y



