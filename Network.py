import torch
import torch.nn as nn
import torch.nn.functional as F

'''
LSTM Network.
We have used EncoderLSTM which defines the Encoder architecture.
LuongDecoder model which defines the decoder architecture.
Attention model which defines the Attention mechanism.
'''
device = "cuda"
class EncoderLSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, nLayers = 1, dropProb = 0):
        super(EncoderLSTM, self).__init__()
        self.hiddenSize = hiddenSize
        self.nLayers = nLayers
        self.embedding = nn.Embedding(inputSize, hiddenSize)
        self.lstm = nn.LSTM(hiddenSize, hiddenSize, nLayers,
                            dropout=dropProb, batch_first=True)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batchSize = 1):
        return (torch.zeros(self.nLayers, batchSize, self.hiddenSize, device=torch.device(self.device)),
                torch.zeros(self.nLayers, batchSize, self.hiddenSize, device=torch.device(self.device)))


class LuongDecoder(nn.Module):
    def __init__(self, hiddenSize, outputSize, attention, nLayers = 1, dropProb=0.1):
        super(LuongDecoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.nLayers = nLayers
        self.dropProb = dropProb

        self.attention = attention
        self.embedding = nn.Embedding(self.outputSize, self.hiddenSize)
        self.dropout = nn.Dropout(self.dropProb)
        self.lstm = nn.LSTM(self.hiddenSize, self.hiddenSize, nLayers)
        self.classifier = nn.Linear(self.hiddenSize*2, self.outputSize)

    def forward(self, inputs, hidden, encoderOutputs):
        embedded = self.embedding(inputs).view(1,1,-1)
        embedded = self.dropout(embedded)

        lstmOut, hidden = self.lstm(embedded, hidden)

        alignmentScore = self.attention(lstmOut, encoderOutputs)
        attnWeights = F.softmax(alignmentScore.view(1,-1), dim=1)
        contextVector = torch.bmm(attnWeights.unsqueeze(0), encoderOutputs)
        output = torch.cat((lstmOut, contextVector), -1)
        output = F.log_softmax(self.classifier(output[0]), dim=1)
        return output, hidden, attnWeights

class Attention(nn.Module):
    def __init__(self, hiddenSize, method = "dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hiddenSize = hiddenSize

        if method == "general":
            self.fc = nn.Linear(hiddenSize, hiddenSize, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hiddenSize, hiddenSize, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hiddenSize))

    def forward(self, decodeHidden, encoderOutputs):
        if self.method == "dot":
            return encoderOutputs.bmm(decodeHidden.view(1,-1,1)).squeeze(-1)

        elif self.method == "general":
            out = self.fc(decodeHidden)
            return encoderOutputs.bmm(out.view(1,-1,1)).squeeze(-1)

        elif self.method == "concat":
            out = torch.tanh(self.fc(decodeHidden+encoderOutputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)

