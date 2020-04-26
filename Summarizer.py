from Network import EncoderLSTM, LuongDecoder, Attention
from DataPreprocessing import Preprocess
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

filePath = "20200325_counsel_chat.csv"
columns = ['questionText', 'answerText']
process = Preprocess(filePath, columns)
process.readData()
process.cleanText()
process.preprocess(0.1)

hiddenSize = 256
encoder = EncoderLSTM(len(process.en_tr_words), hiddenSize).to(torch.device("cuda"))
attn = Attention(hiddenSize, "concat")
decoder = LuongDecoder(hiddenSize, len(process.de_tr_words), attn).to(torch.device("cuda"))

lr = 0.001
encoderOptimizer = optim.Adam(encoder.parameters(), lr=lr)
decoderOptimizer = optim.Adam(decoder.parameters(), lr=lr)


EPOCHS = 10
teacher_probe = 0.5
encoder.train()
decoder.train()
#tk0 = tqdm_notebook(range(1, EPOCHS+1), total = EPOCHS, disable=False)
for epoch in range(EPOCHS):
    avg_loss = 0
    #tk1 = tqdm_notebook(enumerate(process.en_tr_inputs), total=len(process.en_tr_inputs),
    #                    leave=False)
    for i, sentence in enumerate(process.en_tr_inputs):
        loss = 0.
        print("Processing Journal: ", i)
        h = encoder.init_hidden()
        encoderOptimizer.zero_grad()
        decoderOptimizer.zero_grad()
        inp = torch.tensor(sentence).unsqueeze(0).to(torch.device("cuda"))
        encoderOutputs, h = encoder(inp, h)

        decoderInput = torch.tensor([process.en_tr_w2i['_SOS']], device=torch.device("cuda"))
        decoderHidden = h
        output = []
        teacher_forcing = True if random.random()<teacher_probe else False

        for ii in range(len(process.de_tr_inputs[i])):
            decoderOutput, decoderHidden, attnWeights = decoder(decoderInput, decoderHidden, encoderOutputs)
            topVal, topIdx = decoderOutput.topk(1)
            if teacher_forcing:
                decoderInput = torch.tensor([process.de_tr_inputs[i][ii]], device=torch.device("cuda"))
            else:
                decoderInput = torch.tensor([topIdx.item()], device=torch.device("cuda"))
            output.append(topIdx.item())

            loss += F.nll_loss(decoderOutput.view(1,-1),
                               torch.tensor([process.de_tr_inputs[i][ii]],
                                            device=torch.device("cuda")))
        loss.backward()
        encoderOptimizer.step()
        decoderOptimizer.step()
        avg_loss += loss.item()/len(process.en_tr_inputs)
    print("EPOCH: ", epoch, " loss: ", avg_loss)

torch.save({"encoder":encoder.state_dict(),
            "decoder":decoder.state_dict(),
            "e_optimizer":encoderOptimizer.state_dict(),
            "d_optimizer":decoderOptimizer}, "./model.pt")