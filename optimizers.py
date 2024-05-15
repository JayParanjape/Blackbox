import torch
from utils import *

def get_output(model, image, points, boxes, text):
    if not model.use_sam_actual:
        return model(image, points, boxes, text)
    else:
        if len(image.shape)==3:
            return model(image, points, boxes, text)
        else:
            ret = []
            for i in range(image.shape[0]):
                sam_img = image[i].unsqueeze(0)
                if not model.use_sam_auto_mode:
                    sam_points = points[i].unsqueeze(0) if points!=None else None
                else:
                    sam_points = None
                if text!= None:
                    sam_text = [text[i]]
                else:
                    sam_text = None
                sam_out = model(sam_img, sam_points, boxes, sam_text)
                ret.append(sam_out)
            ret = torch.cat(ret,dim=0).to(model.device)
            return ret
                
#adapted from https://github.com/changdaeoh/BlackVIP
def spsa_grad_estimate_bi(model, image, points, boxes, text, label, loss_fn, ck, sp_avg, baseline_expts=False):
        #* repeat k times and average them for stabilizing
        ghats = []
        if baseline_expts:
            w = model.vp
            N_params = w.shape
        else:
            w = torch.nn.utils.parameters_to_vector(model.decoder.parameters())
            N_params = w.shape[0]
        for spk in range(sp_avg):
            #! Bernoulli {-1, 1}
            # perturb = torch.bernoulli(torch.empty(self.N_params).uniform_(0,1)).cuda()
            # perturb[perturb < 1] = -1
            #! Segmented Uniform [-1, 0.5] U [0.5, 1]
            p_side = (torch.rand(N_params).reshape(-1,1) + 1)/2
            samples = torch.cat([p_side,-p_side], dim=1)
            perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)).reshape(-1).to(w.device)
            
            del samples; del p_side

            #* two-side Approximated Numerical Gradient
            if baseline_expts:
                w_r = w + (ck*perturb).reshape(w.shape)
                w_l = w - (ck*perturb).reshape(w.shape)
                model.vp = w_r
            else:
                w_r = w + ck*perturb
                w_l = w - ck*perturb
                torch.nn.utils.vector_to_parameters(w_r, model.decoder.parameters())
            output1 = get_output(model, image, points, boxes, text)
            if baseline_expts:
                model.vp = w_l
            else:
                torch.nn.utils.vector_to_parameters(w_l, model.decoder.parameters())
            output2 = get_output(model, image, points, boxes, text)

            output1 = torch.Tensor(output1).to(label.device)
            output2 = torch.Tensor(output2).to(label.device)
            loss1 = loss_fn.forward(output1, label)
            loss2 = loss_fn.forward(output2, label)

            ghat = (loss1 - loss2)/((2*ck)*perturb)
            if not torch.isnan(torch.mean(ghat)):
                ghats.append(ghat.reshape(1, -1))
        
        if sp_avg == 1: pass
        else: ghat = torch.cat(ghats, dim=0).mean(dim=0) 
        loss = ((loss1+loss2)/2)


        avg_dice = ((dice_coef(label,output1)+
                dice_coef(label, output2))/2).item()

        print("Debug: Magnitude of ghat: ", torch.norm(ghat))
        return ghat, loss, avg_dice