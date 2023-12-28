import torch
from utils import *

def spsa_grad_estimate_bi(model, image, points, boxes, text, label, loss_fn, ck, sp_avg):
        #* repeat k times and average them for stabilizing
        ghats = []
        w = torch.nn.utils.parameters_to_vector(model.decoder.parameters())
        N_params = w.shape[0]
        for spk in range(sp_avg):
            #! Bernoulli {-1, 1}
            # perturb = torch.bernoulli(torch.empty(self.N_params).uniform_(0,1)).cuda()
            # perturb[perturb < 1] = -1
            #! Segmented Uniform [-1, 0.5] U [0.5, 1]
            p_side = (torch.rand(N_params).reshape(-1,1) + 1)/2
            samples = torch.cat([p_side,-p_side], dim=1)
            perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)).reshape(-1).cuda()
            perturb = perturb.to(w.device)
            del samples; del p_side

            #* two-side Approximated Numerical Gradient
            w_r = w + ck*perturb
            w_l = w - ck*perturb
            torch.nn.utils.vector_to_parameters(w_r, model.decoder.parameters())
            output1 = model(image, points, boxes, text)
            torch.nn.utils.vector_to_parameters(w_l, model.decoder.parameters())
            output2 = model(image, points, boxes, text)
            output1 = torch.Tensor(output1).to(label.device)
            output2 = torch.Tensor(output2).to(label.device)
        #     print(f"debug: output shape: {output1.shape} label shape: {label.shape}")
            loss1 = loss_fn.forward(output1, label)
            loss2 = loss_fn.forward(output2, label)

            #* parameter update via estimated gradient
            ghat = (loss1 - loss2)/((2*ck)*perturb)
            ghats.append(ghat.reshape(1, -1))
        if sp_avg == 1: pass
        else: ghat = torch.cat(ghats, dim=0).mean(dim=0) 
        loss = ((loss1+loss2)/2)


        avg_dice = ((dice_coef(label,output1)+
                dice_coef(label, output2))/2).item()

        print("Debug: Magnitude of ghat: ", torch.norm(ghat))
        return ghat, loss, avg_dice