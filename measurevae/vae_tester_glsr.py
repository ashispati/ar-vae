from measurevae.measure_vae import MeasureVAE
from measurevae.vae_tester import VAETester


class VAETesterGLSR(VAETester):
    def __init__(
            self,
            dataset,
            model: MeasureVAE,
            has_reg_loss=False,
            reg_type=None,
            reg_dim=0
    ):
        super(VAETesterGLSR, self).__init__(
            dataset,
            model,
            has_reg_loss,
            reg_type,
            reg_dim
        )
        self.trainer_config += 'GLSR'
        self.model.update_trainer_config(self.trainer_config)
        self.model.load()
        self.model.cuda()
