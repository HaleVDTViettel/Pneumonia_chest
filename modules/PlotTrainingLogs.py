import matplotlib.pyplot as plt
import pytorch_lightning as pl

class PlotTrainingLogsCallback(pl.Callback):
    def __init__(self):
        self.validation_losses = []
        self.training_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss_epoch" in trainer.logged_metrics:
            train_loss = trainer.logged_metrics["train_loss_epoch"].cpu().numpy()
            self.training_losses.append(train_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if "val_loss" in trainer.callback_metrics:
            validation_loss = trainer.callback_metrics["val_loss"].cpu().numpy()
            self.validation_losses.append(validation_loss)            

    def on_fit_end(self, trainer, pl_module):
        plt.plot(self.training_losses, label="Training Loss")
        plt.plot(self.validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()   