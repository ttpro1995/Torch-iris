require 'nn'
require 'torch'
require 'model/SimpleSeqModel'
require 'optim'

local loader = require "iris_loader"

opt = {}
opt.size_in = 4
opt.size_out = 4

-- create model
model = create_seq(opt)

-- criterion
criterion = nn.ClassNLLCriterion()

-- params
params, gradParams = model:getParameters()

-- dataset
dataset = loader.load_data()
x = dataset.inputs
y = dataset.targets


-- feval function
function feval(params)
	gradParams:zero()
	local outputs = model:forward(x)
	local loss = criterion:forward(outputs,y)
	local dloss_doutputs = criterion:backward(outputs,y)
	model:backward(x,dloss_doutputs)
	return loss, gradParams
end

-- Train
optimState = {
	learningRate = 0.01
}

for epoch = 1, 1000 do
	optim.sgd(feval,params,optimState)
end

-- evaluate on dataset

result = model:forward(x)
val, idx = torch.max(result,2) -- find max value in 2 dimemsion
-- idx is class of each sample classified by model

--compare with truth ground
mask = idx:eq(y:long())
acc = mask:sum()/mask:size(1)
print('accurate '..acc)
