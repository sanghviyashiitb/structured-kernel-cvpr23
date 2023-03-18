# Official Code for Structured Kernel Estimation for Photon-Limited Deconvolution  (CVPR 2023)

![Iterative-Scheme-KTN](https://user-images.githubusercontent.com/20774419/226128164-0a98b51b-cfbc-42a9-b32d-8db3ccdedf5c.png)
![ktn-principle](https://user-images.githubusercontent.com/20774419/226128271-2546d404-0ae9-4e35-916d-5009064c5a0a.png)


## Instructions
1. Create a local copy of repository using the following commands
      ```console
      foor@bar:~$ git clone https://github.com/sanghviyashiitb/structured-kernel-cvpr23.git
      foor@bar:~$ cd structured-kernel-cvpr23
      foor@bar:~/structured-kernel-cvpr23$       
      ```
      
2. Download the pretrained models, i.e. denoiser, p4ip, and ktn  into ```model_zoo``` from the link [here](https://drive.google.com/drive/folders/1pzvzZ4Hzt8i6JvuAIaZDjGCjC3i0YX4p?usp=share_link)
      
3. To test the network on levin-data, run the file 
      ```console
      foor@bar:~/structured-kernel-cvpr23$ python3 demo_grayscale.py  
      ```
      ![demo_real](results/demo_grayscale_output.png)
4. 3. To test the network on real-sensor data, run the file 
      ```console
      foor@bar:~/structured-kernel-cvpr23$ python3 demo_real.py  
      ```
      ![demo_real](results/demo_real_output.png)
      
      

 ### Citation
 
 ```
@article{sanghvi2023structured,
  title={Structured Kernel Estimation for Photon-Limited Deconvolution},
  author={Sanghvi, Yash and Mao, Zhiyuan and Chan, Stanley H},
  journal={arXiv preprint arXiv:2303.03472},
  year={2023}
}
 ```

Feel free to ask your questions/share your feedback at sanghviyash95@gmail.com
