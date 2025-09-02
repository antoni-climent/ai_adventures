import torch
import torch.nn as nn
import sys
import time

def inference(model, N, T, z):
    x, y = get_data_batch(1)
    output = model.infer(z, x, 1, N, T)
    output = [1 if o > 0.5 else 0 for o in output]
    return [int(el) for el in y.tolist()[0]], output


def load_model(path):
    checkpoint = torch.load(path, map_location="cuda")

    model = HRM(
        d_model=int(checkpoint["d_model"]),
        nhead=int(checkpoint["nhead"]),
        num_layers=int(checkpoint["num_layers"]),
        out_size=int(checkpoint["out_size"]),
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def save_model(model, d_model, nhead, num_layers, out_size, N, T, z_init,  path):
    torch.save(
        {
            "model_state": model.state_dict(),
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "out_size": out_size,
            "N": N,
            "T": T,
            "z_init": z_init, 
        },
        path,
    )


def get_data_batch(batch_size=64, device="cuda"):
    n1 = torch.randint(
        0, 256, (batch_size,), device=device, dtype=torch.int16
    )  # It will use 8 bits
    n2 = torch.randint(0, 256, (batch_size,), device=device, dtype=torch.int16)
    s = n1 + n2

    # n1 = torch.tensor([7,1], device=device)
    # n2 = torch.tensor([4,5], device=device)
    # s = torch.tensor([11,6], device=device)
    # n1 = torch.tensor([7], device=device)
    # n2 = torch.tensor([4], device=device)
    # s = torch.tensor([11], device=device)

    shift8 = torch.arange(7, -1, -1, device=device, dtype=torch.int16)
    shift9 = torch.arange(8, -1, -1, device=device, dtype=torch.int16)

    b1 = ((n1.unsqueeze(1).bitwise_right_shift(shift8)) & 1).to(torch.long)
    b2 = ((n2.unsqueeze(1).bitwise_right_shift(shift8)) & 1).to(torch.long)
    y = ((s.unsqueeze(1).bitwise_right_shift(shift9)) & 1).to(torch.float32)

    x = torch.cat([b1, b2], dim=1)

    return x, y


class HRM(nn.Module):
    def __init__(self, d_model, nhead, num_layers, out_size, device="cuda"):
        super(HRM, self).__init__()
        self.num_embed = nn.Embedding(2, d_model, device=device)
        self.pos_embed = nn.Embedding(16, d_model, device=device)
        self.encoderLayerL = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, device=device
        )
        self.encoderLayerH = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, device=device
        )

        self.moduleL = nn.TransformerEncoder(
            encoder_layer=self.encoderLayerL, num_layers=num_layers
        )
        self.moduleH = nn.TransformerEncoder(
            encoder_layer=self.encoderLayerH, num_layers=num_layers
        )

        self.moduleO = nn.Linear(
            in_features=d_model, out_features=out_size, device=device
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, z, x, batch_size, N, T, device="cuda"):
        zH = z[0]
        zL = z[1]
        n_emb = self.num_embed(x)  # (batch_size, 16, d_model)
        pos_emb = (
            self.pos_embed(torch.arange(0, 16, device=device))
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )  # (batch_size, 16, d_model)
        final_emb = n_emb + pos_emb
 
        with torch.no_grad(): 
            for _i in range(N * T - 1):
                a = final_emb.shape 
                b = zH.shape
                c = zL.shape
                inputL = final_emb + zH + zL
                # print("inputL: ", inputL.size())
                zL = self.moduleL(inputL)

                if (_i + 1) % T == 0:
                    inputH = zH + zL
                    zH = self.moduleH(inputH)

        # 1-step grad
        inputL = final_emb + zH + zL
        zL = self.moduleL(inputL)

        inputH = zH + zL
        zH = self.moduleH(inputH)  # (1, 16, d_model)
        pool = zH.mean(dim=1).squeeze(0)  # (d_model)
        output = self.moduleO(pool)

        return [zH, zL], output

    def train_net(self, z, optimizer, batch_size, num_samples, n_supervision, N, T):
        for sample in range(num_samples):
            loss_list = []
            t1 = time.time()
            x, y_true = get_data_batch(batch_size)
            for _ in range(n_supervision):
                z, y_hat = self.forward(z, x, batch_size, N=N, T=T)
                loss = self.criterion(y_hat, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Prevent the gradients to influence next round
                z[0] = z[0].detach()
                z[1] = z[1].detach()
                loss_list.append(loss)
            print(f"[{sample}/{num_samples}] -> Loss: {sum(loss_list) / len(loss_list):.5f}, with time: {time.time() - t1:.5f}s")
            
            # Save checkpoint
            if sample%25 == 0:
                torch.save(self.state_dict(), f"checkpoint{sample}.pth")


    def infer(self, z, x, batch_size, N, T):
        _, output = self.forward(z, x, batch_size, N=N, T=T, device="cuda")
        return torch.sigmoid(output)


if __name__ == "__main__":

    # Hiperparameter definition
    model_name = "hrm_model.pth"
    d_model = 512
    batch_size = 512
    nhead = 8
    num_layers = 8
    out_size = 9
    N = 2
    T = 2
    n_supervision = 4
    input_size = 16
    device = "cuda"
    is_training = True
    num_samples = 3000
    torch.manual_seed(42)

    # Choose between training and inference
    if is_training:
        model = HRM(d_model=d_model, nhead=nhead, num_layers=num_layers, out_size=out_size)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

        z_init = torch.empty(d_model, device=device)
        nn.init.trunc_normal_(z_init, mean=0, std=1.0, a=-2.0, b=2.0)
        z_init = z_init.to(torch.float32)
        z_save = z_init.clone() # For later storage with model weights
        z_init = z_init.unsqueeze(0).repeat(batch_size,input_size,1)
        z = [z_init.clone(), z_init.clone()]
        model.train_net(z, optimizer, batch_size, num_samples=num_samples, n_supervision=n_supervision, N=N, T=T)
        save_model(model, d_model, nhead, num_layers, out_size, N, T, z_save, model_name)
    else:
        model = load_model(model_name)

        model_info = torch.load(model_name, map_location=device)
        z_init = model_info["z_init"]
        z_init = z_init.repeat(1,input_size,1)

        z = [z_init.clone(), z_init.clone()]
        
        for _ in range(20):
            y_true, y_predict = inference(model, N, T, z)
            print("Y_true: ", y_true)
            print("Y_pred: ", y_predict)
            print("-----------------")




