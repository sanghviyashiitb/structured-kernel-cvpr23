import torch

def shock(I0,iter,dt):
    h = 1

    I = I0

    ss = I0.size()

    for i in range(0,iter):
        Itempmx = I.clone()
        Itempmx[:,:,:,1:ss[3]] = Itempmx[:,:,:,0:ss[3]-1].clone()
        I_mx = I - Itempmx

        Itemppx = I.clone()
        Itemppx[:,:,:,0:ss[3]-1] = Itemppx[:,:,:,1:ss[3]].clone()
        I_px = Itemppx - I

        Itempmy = I.clone()
        Itempmy[:,:,1:ss[2],:] = Itempmy[:,:,0:ss[2] - 1,:].clone()
        I_my = I - Itempmy

        Itemppy = I.clone()
        Itemppy[:,:,0:ss[2] - 1,:] = Itemppy[:,:,1:ss[2],:].clone()
        I_py = Itemppy - I

        I_x = (I_mx + I_px) / 2
        I_y = (I_my + I_py) / 2

        #minmod operator
        Dx = torch.minimum(torch.abs(I_mx), torch.abs(I_px))
        Dx[I_mx*I_px < 0] = 0

        Dy = torch.minimum(torch.abs(I_my), torch.abs(I_py))
        Dy[I_my * I_py < 0] = 0

        I_xx = Itemppx + Itempmx - 2*I
        I_yy = Itemppy + Itempmy - 2*I

        Itempxy1 = I_x.clone()
        Itempxy1[:,:,0: ss[2] - 1,:] = Itempxy1[:,:,1:ss[2], :].clone()

        Itempxy2 = I_x.clone()
        Itempxy2[:,:,1:ss[2],:] = Itempxy2[:,:,0:ss[2] - 1,:].clone()

        I_xy = (Itempxy1 - Itempxy2)/2

        #compute flow
        a_grad_I = torch.sqrt(Dx**2 + Dy**2)
        dl = 1e-8
        I_nn = I_xx * (torch.abs(I_x)**2) + 2*I_xy * I_x * I_y + I_yy * (torch.abs(I_y)**2)
        I_nn = I_nn/((torch.abs(I_x)**2) + (torch.abs(I_y)**2) + dl)

        I_ee = I_xx * (torch.abs(I_y)**2) - 2*I_xy * I_x * I_y + I_yy * (torch.abs(I_x)**2)
        I_ee = I_ee / ((torch.abs(I_x) ** 2) + (torch.abs(I_y) ** 2) + dl)

        a2_grad_I = torch.abs(I_x) + torch.abs(I_y)

        I_nn[a2_grad_I == 0] = I_xx[a2_grad_I == 0]
        I_ee[a2_grad_I == 0] = I_yy[a2_grad_I == 0]

        I_t = -torch.sign(I_nn)*a_grad_I/h

        I = I + dt*I_t

    return I




