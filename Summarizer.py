from Network import EncoderLSTM, LuongDecoder, Attention
from DataPreprocessing import Preprocess
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import Network as network
'''
Loading the data and preprocessing it.
Training Network.
Saving the Network.
'''

filePath = "20200325_counsel_chat.csv"
columns = ['questionText', 'answerText']
process = Preprocess(filePath, columns)
process.readData()
process.cleanText()
process.preprocess(0.1)

hiddenSize = 256
encoder = EncoderLSTM(len(process.en_tr_words), hiddenSize).to(torch.device(network.device))
attn = Attention(hiddenSize, "concat")
decoder = LuongDecoder(hiddenSize, len(process.de_tr_words), attn).to(torch.device(network.device))

lr = 0.001
encoderOptimizer = optim.Adam(encoder.parameters(), lr=lr)
decoderOptimizer = optim.Adam(decoder.parameters(), lr=lr)


EPOCHS = 10
teacher_probe = 0.5
encoder.train()
decoder.train()
for epoch in range(EPOCHS):
    avg_loss = 0
    for i, sentence in enumerate(process.en_tr_inputs):
        loss = 0.
        h = encoder.init_hidden()
        encoderOptimizer.zero_grad()
        decoderOptimizer.zero_grad()
        inp = torch.tensor(sentence).unsqueeze(0).to(torch.device(network.device))
        encoderOutputs, h = encoder(inp, h)

        decoderInput = torch.tensor([process.en_tr_w2i['_SOS']], device=torch.device(network.device))
        decoderHidden = h
        output = []
        teacher_forcing = True if random.random()<teacher_probe else False

        for ii in range(len(process.de_tr_inputs[i])):
            decoderOutput, decoderHidden, attnWeights = decoder(decoderInput, decoderHidden, encoderOutputs)
            topVal, topIdx = decoderOutput.topk(1)
            if teacher_forcing:
                decoderInput = torch.tensor([process.de_tr_inputs[i][ii]], device=torch.device(network.device))
            else:
                decoderInput = torch.tensor([topIdx.item()], device=torch.device(network.device))
            output.append(topIdx.item())

            loss += F.nll_loss(decoderOutput.view(1,-1),
                               torch.tensor([process.de_tr_inputs[i][ii]],
                                            device=torch.device(network.device)))
        loss.backward()
        encoderOptimizer.step()
        decoderOptimizer.step()
        avg_loss += loss.item()/len(process.en_tr_inputs)
    print("EPOCH: ", epoch, " loss: ", avg_loss)

torch.save({"encoder":encoder.state_dict(),
            "decoder":decoder.state_dict(),
            "e_optimizer":encoderOptimizer.state_dict(),
            "d_optimizer":decoderOptimizer}, "./model.pt")
