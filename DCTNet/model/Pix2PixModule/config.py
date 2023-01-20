class Params:
    def __init__(self):
       
        self.name = 'Pix2Pix'
        self.network_name = 'TTN'
    
        self.pretrain_path = None
        
        self.lr = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.99

        self.use_exp = True
        self.lambda_surface = 2.0
        self.lambda_texture = 2.0
        self.lambda_content = 200
        self.lambda_tv = 1e4
       
        self.lambda_exp = 1.0

        self.infer_batch_size = 2
        self.wandb_project_name = 'Webtoonme (DCT-Net)'

