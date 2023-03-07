import numpy as np
from scipy.io import loadmat


def p4ip_wrapper_pad(y, k, M, p4ip):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	H, W = np.shape(y)[0], np.shape(y)[1]
	H1, W1 =  np.int32(H/4), np.int32(W/4)  
	y_pad = np.pad(y, ((H1,H1),(W1,W1)), mode='symmetric')


	Ht = img_to_tens(k).to(device).float()	
	yt = img_to_tens(y_pad).to(device)
	Mt = scalar_to_tens(M).to(device)

	with torch.no_grad():
		x_rec_list = p4ip(yt, Ht, Mt)
	x_rec = x_rec_list[-1]
	x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)
	x_out = x_rec[0,0,:,:]
	x_out = x_out[H1:H1+H, W1:W1+W]
	return x_out

def estimate_err_nb(y, yn, k, M, p4ip):
	H, W = np.shape(y); H1, W1 = H//4, W//4
	
	x_out = p4ip_wrapper_pad(y, k, M, p4ip )
	x_out_pad = np.pad(x_out, ((H1,H1),(W1,W1)), mode='symmetric')
	_, fft_k = psf2otf(k, [H+2*H1, W+2*W1])
	y_hat_pad = np.clip(np.real(ifft2(fft2(x_out_pad)*fft_k)),0,1)
	y_hat = y_hat_pad[H1:H1+H, W1:W+W1]
	err = np.mean((y_hat-yn)**2)

	return err
	
# Do a grid-search
def grid_search(y, yn, M, blind_deconv)
	struct = loadmat('data/rectilinear_kernels.mat')
	kernel_list = struct['kernel_list']
	best_err = np.inf
		for i1 in range(20):
			for j1 in range(18):
				k0 = np.reshape(kernel_list[i1,j1,:], [64,64])
				k0 = k0/np.sum(np.ravel(k0))
				err = estimate_err_nb(y, yn, k0, M, blind_deconv.non_blind_solver)
				if err < best_err:
					best_err = err
					best_k = k0
			
	return best_k