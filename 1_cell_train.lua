require 'torch'
require 'nn'
require 'nngraph'
require 'model/SimpleRNN'
require 'optim'

local loader = require "iris_loader"

opt = {}
opt.input_size = 4
opt.rnn_size = 5

-- create model
model = RNN.create(opt)

--
-- criterion
criterion = nn.ClassNLLCriterion()

-- params
params, gradParams = model:getParameters()

-- dataset
dataset = loader.load_data()
x = dataset.inputs
h_0= torch.Tensor(150,opt.rnn_size):zero()
input = {x,h_0}
y = dataset.targets

function feval(params)
  gradParams:zero()
  local output = model:forward(input)
  local h = output[1]
  local predict = output[2]
  local lost = criterion:forward(predict, y)
  local dloss_doutput = criterion:backward(predict, y)
  model:backward(input, dloss_doutput)
  return params, gradParams
end

-- Train
optimState = {
	learningRate = 0.01
}

for epoch = 1, 1000 do
	optim.sgd(feval,params,optimState)
end
