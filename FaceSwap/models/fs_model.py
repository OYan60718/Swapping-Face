import torch

from .base_model import BaseModel
from .fs_networks_fix import Generator_Adain_Upsample

from .projected_discriminator import ProjectedDiscriminator


class fsModel(BaseModel):
    def name(self):
        return 'fsModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep)
        self.netG.cuda()

        netArc_checkpoint = torch.load(opt.Arc_path, map_location=torch.device("cpu"), weights_only=False)
        self.netArc = netArc_checkpoint.cuda()
        self.netArc.eval()
        self.netArc.requires_grad_(False)

        if not self.isTrain:
            self.load_network(self.netG, 'G', opt.which_epoch, opt.checkpoints_dir)
            return

        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        self.netD.cuda()

        self.optimizer_G = torch.optim.Adam(
            list(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8
        )

        if opt.continue_train:
            pretrained_path = opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)

        torch.cuda.empty_cache()

    def cosin_metric(self, x1, x2):
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        self.save_optim(self.optimizer_G, 'G', which_epoch)
        self.save_optim(self.optimizer_D, 'D', which_epoch)