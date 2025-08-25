import torch
import torch.nn as nn

def get_data_sample():
    n1 = torch.randint(0, 255, ()) # It will use 8 bits
    n2 = torch.randint(0, 255, ())
    s = n1 + n2
    
    n1 = [int(bit) for bit in bin(n1)[2:]]
    n2 = [int(bit) for bit in bin(n2)[2:]]
    s = [int(bit) for bit in bin(s)[2:]]
    
    # Make them the same size 
    n1 = (8 - len(n1))*[0] + n1 
    n2 = (8 - len(n2))*[0] + n2 
    s = (9 - len(s))*[0] + s # To represent up to 510 we need 9 bits
    
    return torch.cat((torch.tensor(n1),torch.tensor(n2))), torch.tensor(s)

class HRM(nn.Module):
    
    def __init__(self, d_model, nhead, num_layers, out_size):
        super(HRM, self).__init__()
        self.num_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Embedding(16, d_model)
        self.encoderLayerL = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoderLayerH = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

        self.moduleL = nn.TransformerEncoder(encoder_layer=self.encoderLayerL, num_layers=num_layers)
        self.moduleH = nn.TransformerEncoder(encoder_layer=self.encoderLayerH, num_layers=num_layers)

        self.moduleO = nn.Linear(in_features=d_model, out_features=out_size)

    def forward(self, z, x, N=2, T=2):
        sigm = nn.Sigmoid()

        zH = z[0]
        zL = z[1]
        n_emb = self.num_embed(x)
        pos_emb = self.pos_embed(torch.arange(0,16))
        print(n_emb.size())
        print(pos_emb.size())
        final_emb = torch.add(n_emb, pos_emb) # shape: (8,d_model)
        with torch.no_grad():
            for _i in range(N*T - 1):
                print("final_emb:", final_emb.size())
                print("zH", zH.size())
                print("zL", zL.size())
                # zH = zH.view(1,-1)
                # zL = zL.view(1,-1)
                inputL = final_emb + zH + zL
                print("inputL", inputL.size())
                zL = self.moduleL(inputL)
                
                if (_i + 1)%T == 0:
                    inputH = zH + zL
                    zH = self.moduleH(inputH)

        # 1-step grad 
        inputL = final_emb + zH + zL
        zL = self.moduleL(inputL)
            
        inputH = zH + zL
        zH = self.moduleH(inputH)
        pool = zH.mean(dim=0)
        print("pool:", pool.size())
        output_head = self.moduleO(pool)

        print("output_head:", output_head)
        output = sigm(output_head)

        return [zH, zL], output

    def train_net(self, z_init, optimizer, epochs, N_supervision):
        bce= nn.BCELoss()
        
        for _ in range(epochs):
            x, y_true = get_data_sample()
            z = [z_init, z_init]
            for _ in range(N_supervision):
                z, y_hat = self.forward(z, x) 
                loss = bce(y_hat, y_true)
                optimizer.zero_grad()
                loss.bachward()
                optimizer.step()
                
                # Prevent the gradients to influence next round
                z[0] = z[0].detach()
                z[1] = z[1].detach()
                print(loss)
                


if __name__ == "__main__":
    d_model = 512
    model = HRM(d_model, 4, 8, 9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Training 
    z = torch.empty(d_model, dtype=torch.float16)
    nn.init.trunc_normal_(z, mean=0, std=1.0, a=-2.0, b=2.0)
    model.train_net(z, optimizer, epochs=10000, N_supervision=10)
