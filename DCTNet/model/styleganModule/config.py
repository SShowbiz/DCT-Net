class Params:
    def __init__(self):
        
        self.name = 'StyleGAN'
        self.network_name = 'CCN'
        
        self.g_reg_every = 4 
        self.d_reg_every = 16
        self.D_steps_pre_G = 1
        self.latent = 512
        self.n_mlp =8
        self.channel_multiplier =2
        self.size =1024
        self.mixing =0.9
        self.inject_index =4
        self.n_sample =8

        self.lambda_gan =1.0
        self.lambda_id =1.0
        self.path_regularize =2.0
        self.path_batch_shrink = 2
        self.r1 =10.0
        
        self.interval_steps = 100
        self.interval_train = False

        self.infer_batch_size = 1
        self.mx_gen_iters = 10000
        self.wandb_project_name = 'Webtoonme (DCT-Net)'

        self.truncation = 0.5
        self.numvec_for_truncation = 4096
