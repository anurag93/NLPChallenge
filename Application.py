from Network import EncoderLSTM, LuongDecoder, Attention
from DataPreprocessing import Preprocess
import torch

'''
Application to create and load the dictionaries. 
I need to come up with an architecture ti remove its dependency on Preprocess class.
'''
filePath = "20200325_counsel_chat.csv"
columns = ['questionText', 'questionTitle']
process = Preprocess(filePath, columns)
process.readData()
process.cleanText()
process.preprocess(0.1)
process.preprocessMaps()

'''
Configuring the Network parameters.
Loading th model.
Prompting the user for the journal.
Starting the inference network and generating the reflection.  
'''

hiddenSize = 256
encoder = EncoderLSTM(len(process.en_words), hiddenSize).to(torch.device(EncoderLSTM.device))
attn = Attention(hiddenSize, "concat")
decoder = LuongDecoder(hiddenSize, len(process.de_words), attn).to(torch.device(EncoderLSTM.device))

checkpoint = torch.load('model.pt')

encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()
h = encoder.init_hidden()


question = input("Enter your question: ")
process.encodeText(question)

inp = torch.tensor(process.textInput).unsqueeze(0).to(torch.device(EncoderLSTM.device))
encoder_outputs, h = encoder(inp,h)

decoder_input = torch.tensor([process.en_w2i['_SOS']],device=torch.device(EncoderLSTM.device))
decoder_hidden = h
output = []
attentions = []
count = 0
while True and count<40:
  decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
  _, top_index = decoder_output.topk(1)
  decoder_input = torch.tensor([top_index.item()],device=torch.device(EncoderLSTM.device))
  if top_index.item() == process.de_w2i["_EOS"]:
    break
  output.append(top_index.item())
  count = count+1
  attentions.append(attn_weights.squeeze().cpu().detach().numpy())

reflection = " ".join([process.de_i2w[x] for x in output])
reply = process.fetchReply(reflection)
print("Youper: ", reply)