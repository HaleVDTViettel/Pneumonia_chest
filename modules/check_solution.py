import pytorch_lightning as pl

from .PneumoniaData import PneumoniaDataModule
from .PneumoniaModel import PneumoniaModel
from .InfoPrinter import InfoPrinterCallback
from .PlotTestConfusionMatrix import PlotTestConfusionMatrixCallback
from .PlotTrainingLogs import PlotTrainingLogsCallback
 
def check_solution(h, verbose):
    pneumonia_data = PneumoniaDataModule(h, "Pediatric_Chest_X_ray_Pneumonia/")
    pneumonia_model = PneumoniaModel(h)

    # Callbacks
    info_printer = InfoPrinterCallback()

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=h["early_stopping_patience"],
        verbose=True,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="model_checkpoints",
        monitor="val_loss",
        verbose=True,
    )

    callbacks = [info_printer, 
                #  early_stopping, 
                 checkpoint_callback]
    if (verbose):
        callbacks.append(PlotTestConfusionMatrixCallback())
        callbacks.append(PlotTrainingLogsCallback())

    trainer = pl.Trainer(
        max_epochs=h["num_epochs"],
        accelerator="auto",
        callbacks=callbacks,
        log_every_n_steps=1,
        fast_dev_run=False
    )

    trainer.fit(pneumonia_model, datamodule=pneumonia_data)

    if (h["use_best_checkpoint"]):
        #Debug lines
        trainer.test(pneumonia_model, datamodule=pneumonia_data)
        print(f"Last: F1= {pneumonia_model.test_f1:.4f}, Acc= {pneumonia_model.test_acc:.4f}")

        best_model_path = checkpoint_callback.best_model_path
        best_model = PneumoniaModel.load_from_checkpoint(best_model_path, h=h)
        pneumonia_model = best_model

    trainer.test(pneumonia_model, datamodule=pneumonia_data)
    print(f"Best: F1= {pneumonia_model.test_f1:.4f}, Acc= {pneumonia_model.test_acc:.4f}")


    return pneumonia_model.test_f1, pneumonia_model.test_acc